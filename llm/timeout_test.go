package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// stallServer streams one Anthropic text delta, then stays silent until the
// test ends (or the client disconnects).
func stallServer(t *testing.T, gap time.Duration) *httptest.Server {
	done := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "text/event-stream")
		f := w.(http.Flusher)
		fmt.Fprint(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n")
		fmt.Fprint(w, "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		fmt.Fprint(w, "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n")
		f.Flush()
		select {
		case <-time.After(gap):
			fmt.Fprint(w, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
			f.Flush()
		case <-r.Context().Done():
		case <-done:
		}
	}))
	t.Cleanup(func() { close(done); srv.Close() })
	return srv
}

func streamOnce(t *testing.T, srv *httptest.Server, idle time.Duration) (text string, err error, took time.Duration) {
	p, perr := NewProvider(Config{Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Model: "m", MaxRetries: -1, StreamIdleTimeout: idle})
	if perr != nil {
		t.Fatal(perr)
	}
	start := time.Now()
	for ev, e := range p.Stream(context.Background(), CompletionRequest{Messages: []Message{UserText("x")}}) {
		if e != nil {
			return text, e, time.Since(start)
		}
		text += ev.Text
	}
	return text, nil, time.Since(start)
}

// A stream that goes silent mid-response ends with ErrStreamIdle after the
// idle window, instead of blocking until the caller's context dies.
func TestStreamIdleTimeoutEndsStalledStream(t *testing.T) {
	srv := stallServer(t, time.Minute)
	text, err, took := streamOnce(t, srv, 300*time.Millisecond)
	if !errors.Is(err, ErrStreamIdle) {
		t.Fatalf("err = %v, want ErrStreamIdle", err)
	}
	if text != "hi" {
		t.Fatalf("text before the stall = %q", text)
	}
	if took > 5*time.Second {
		t.Fatalf("idle timeout took %s", took)
	}
}

// A gap shorter than the window is a slow but live stream: it must complete.
func TestStreamIdleTimeoutSparesLiveStream(t *testing.T) {
	srv := stallServer(t, 200*time.Millisecond)
	if _, err, _ := streamOnce(t, srv, 2*time.Second); err != nil {
		t.Fatalf("live stream cut: %v", err)
	}
}

// Off by default: zero keeps the old behaviour (no wrapper).
func TestStreamIdleTimeoutOffByDefault(t *testing.T) {
	srv := stallServer(t, 200*time.Millisecond)
	if _, err, _ := streamOnce(t, srv, 0); err != nil {
		t.Fatalf("err = %v", err)
	}
}
