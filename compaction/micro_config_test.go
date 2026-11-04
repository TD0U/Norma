package compaction

import (
	"fmt"
	"testing"

	"github.com/TD0U/Norma/llm"
)

// readHistory builds n Read calls, each answered with a non-empty result.
func readHistory(n int, tool string) []llm.Message {
	var msgs []llm.Message
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("t%d", i)
		msgs = append(msgs,
			llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Type: llm.BlockToolUse, ID: id, Name: tool}}},
			llm.Message{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.ToolResultText(id, "body", false)}},
		)
	}
	return msgs
}

func countIntact(msgs []llm.Message) int {
	n := 0
	for _, m := range msgs {
		for _, b := range m.Content {
			if b.Type == llm.BlockToolResult && !isCleared(b.Content) {
				n++
			}
		}
	}
	return n
}

func TestMicroKeepRecentConfigurable(t *testing.T) {
	msgs := readHistory(20, "Read")
	for _, tc := range []struct{ keep, want int }{{0, defaultMicroKeepRecent}, {15, 15}, {30, 20}} {
		c := New(Config{MicroKeepRecent: tc.keep}, nil)
		out, _ := c.microCompact(msgs)
		if got := countIntact(out); got != tc.want {
			t.Errorf("MicroKeepRecent=%d: %d intact, want %d", tc.keep, got, tc.want)
		}
	}
}

func TestCompactableToolsConfigurable(t *testing.T) {
	// LS is not compactable by default, so nothing is cleared.
	if out, changed := New(Config{}, nil).microCompact(readHistory(10, "LS")); changed || countIntact(out) != 10 {
		t.Fatal("LS cleared under the default allow-list")
	}
	c := New(Config{CompactableTools: map[string]bool{"LS": true}}, nil)
	if out, _ := c.microCompact(readHistory(10, "LS")); countIntact(out) != defaultMicroKeepRecent {
		t.Fatal("custom allow-list not applied")
	}
	// A non-nil empty map disables clearing entirely.
	off := New(Config{CompactableTools: map[string]bool{}}, nil)
	if _, changed := off.microCompact(readHistory(10, "Read")); changed {
		t.Fatal("empty allow-list still cleared results")
	}
}
