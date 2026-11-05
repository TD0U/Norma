package tool

import (
	"context"
	"encoding/json"
	"testing"
)

// Set restores silently; TodoWrite notifies OnChange with a detached snapshot.
func TestTodoStoreSetAndOnChange(t *testing.T) {
	s := NewTodoStore()
	var got [][]Todo
	s.OnChange(func(items []Todo) { got = append(got, items) })

	s.Set([]Todo{{Content: "restored", Status: TodoPending}})
	if len(got) != 0 || len(s.List()) != 1 {
		t.Fatalf("Set notified or lost items: got=%v list=%v", got, s.List())
	}
	in := json.RawMessage(`{"todos":[{"content":"a","status":"in_progress"},{"content":"b","status":"pending"}]}`)
	if _, err := s.Tool().Call(context.Background(), in, &ToolContext{}); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || len(got[0]) != 2 || got[0][0].Content != "a" {
		t.Fatalf("OnChange = %v", got)
	}
	got[0][0].Content = "mutated"
	if s.List()[0].Content != "a" {
		t.Fatal("OnChange snapshot aliases the store")
	}
}
