package harness

import (
	"testing"

	"github.com/TD0U/Norma/llm"
	"github.com/TD0U/Norma/permission"
	"github.com/TD0U/Norma/tool"
)

// A queued message lands after the tool results, before the next model round,
// and the tool call still runs (nothing is blocked to deliver it).
func TestPendingMessageAfterToolTurn(t *testing.T) {
	m := &scriptedModel{turns: [][]llm.StreamEvent{
		toolTurn("c1", "LS", `{"path":"."}`),
		textTurn("done"),
	}}
	queue := []string{"steer: look at auth/ instead"}
	in := QueryInput{
		Messages:       []llm.Message{llm.UserText("go")},
		Tools:          tool.NewRegistry(tool.NewLS()),
		PermissionMode: permission.ModeBypass,
		WorkingDir:     t.TempDir(),
		PendingMessages: func() []string {
			q := queue
			queue = nil
			return q
		},
	}
	term := drain(t, in, QueryDeps{CallModel: m.call})
	// user, assistant(tool_use), user(tool_result), user(steer), assistant(text)
	if len(term.Messages) != 5 {
		t.Fatalf("messages=%d: %+v", len(term.Messages), term.Messages)
	}
	if res := term.Messages[2].Content[0]; res.ToolUseID != "c1" || res.IsError {
		t.Fatalf("tool call was not executed: %+v", res)
	}
	if got := term.Messages[3].Text(); got != "steer: look at auth/ instead" {
		t.Fatalf("steer message = %q", got)
	}
	if m.calls != 2 || term.Reason != ReasonCompleted {
		t.Fatalf("calls=%d reason=%s", m.calls, term.Reason)
	}
}

// A message queued during a final (no-tool) turn keeps the run going one more
// round instead of being dropped.
func TestPendingMessageKeepsFinalTurnGoing(t *testing.T) {
	m := &scriptedModel{turns: [][]llm.StreamEvent{
		textTurn("I think I'm done"),
		textTurn("ok, rechecked"),
	}}
	delivered := false
	in := QueryInput{
		Messages:       []llm.Message{llm.UserText("go")},
		Tools:          tool.NewRegistry(),
		PermissionMode: permission.ModeBypass,
		WorkingDir:     t.TempDir(),
		PendingMessages: func() []string {
			if delivered {
				return nil
			}
			delivered = true
			return []string{"not yet — recheck the upload path"}
		},
	}
	term := drain(t, in, QueryDeps{CallModel: m.call})
	if m.calls != 2 || term.Text != "ok, rechecked" {
		t.Fatalf("calls=%d text=%q", m.calls, term.Text)
	}
}
