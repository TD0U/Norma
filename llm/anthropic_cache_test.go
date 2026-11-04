package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func anthropicBody(t *testing.T, cache bool, msgs []Message) string {
	t.Helper()
	p := &anthropicProvider{cfg: Config{Model: "m", CacheMessages: cache}}
	raw, err := p.buildBody(CompletionRequest{Messages: msgs}, true)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func TestAnthropicCacheMessagesMarksTail(t *testing.T) {
	msgs := []Message{
		UserText("first"),
		{Role: RoleAssistant, Content: []ContentBlock{{Type: BlockToolUse, ID: "t1", Name: "Read", Input: json.RawMessage(`{}`)}}},
		{Role: RoleUser, Content: []ContentBlock{ToolResultText("t1", "a", false), ToolResultText("t1", "b", false)}},
	}
	off := anthropicBody(t, false, msgs)
	if strings.Contains(off, "cache_control") {
		t.Fatalf("cache_control sent while off: %s", off)
	}
	on := anthropicBody(t, true, msgs)
	if n := strings.Count(on, "cache_control"); n != 1 {
		t.Fatalf("want exactly 1 breakpoint, got %d: %s", n, on)
	}
	var body struct {
		Messages []struct {
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte(on), &body); err != nil {
		t.Fatal(err)
	}
	last := body.Messages[len(body.Messages)-1].Content
	if _, ok := last[len(last)-1]["cache_control"]; !ok {
		t.Fatalf("breakpoint not on the last block: %s", on)
	}
	// History is not mutated: a second build without caching is clean.
	if strings.Contains(anthropicBody(t, false, msgs), "cache_control") {
		t.Fatal("caller history was mutated")
	}
}
