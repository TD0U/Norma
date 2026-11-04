package harness

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/TD0U/Norma/permission"
	"github.com/TD0U/Norma/tool"
)

// A panicking tool must come back as an error result, not crash the process.
// Tools run on the streaming executor's own goroutines, where a panic cannot be
// recovered by whoever started the run.
func TestExecOneRecoversToolPanic(t *testing.T) {
	boom := tool.Build(tool.Spec{
		Name:   "Boom",
		Schema: map[string]any{"type": "object"},
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Decision{Behavior: permission.Allow}
		},
		Run: func(context.Context, json.RawMessage, *tool.ToolContext) (tool.Result, error) {
			var m map[string]int
			m["x"] = 1 // nil map write
			return tool.Result{}, nil
		},
	})
	res := execWith(t, boom, `{}`)
	if !res.IsError {
		t.Fatal("panicking tool produced a success result")
	}
	if txt := res.Content[0].Text; !strings.Contains(txt, "Boom panicked") || !strings.Contains(txt, "nil map") {
		t.Fatalf("result = %q, want the panic surfaced as an error", txt)
	}
}
