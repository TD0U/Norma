package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

const (
	frameStart   = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3}}}\n\n"
	frameOverld  = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"
	frameBlock   = "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
	frameText    = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n"
	frameStopped = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
)

// sequenceServer serves bodies[i] on the i-th request (the last one repeats).
func sequenceServer(t *testing.T, hits *atomic.Int32, bodies ...string) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		i := int(hits.Add(1)) - 1
		if i >= len(bodies) {
			i = len(bodies) - 1
		}
		w.Header().Set("content-type", "text/event-stream")
		fmt.Fprint(w, bodies[i])
	}))
	t.Cleanup(srv.Close)
	return srv
}

func collectStream(t *testing.T, cfg Config) ([]StreamEvent, error) {
	t.Helper()
	p, err := NewProvider(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var evs []StreamEvent
	for ev, err := range p.Stream(context.Background(), CompletionRequest{Messages: []Message{UserText("x")}}) {
		if err != nil {
			return evs, err
		}
		evs = append(evs, ev)
	}
	return evs, nil
}

func count(evs []StreamEvent, typ StreamEventType) int {
	n := 0
	for _, e := range evs {
		if e.Type == typ {
			n++
		}
	}
	return n
}

func replayCfg(url string, retries int) Config {
	return Config{Format: FormatAnthropic, BaseURL: url, APIKey: "k", Model: "m",
		StreamReplayRetries: retries, StreamReplayInterval: time.Millisecond}
}

// A failure before any output is replayed, and the caller sees one clean
// stream: a single message_start, no trace of the failed attempt.
func TestReplayBeforeCommit(t *testing.T) {
	var hits atomic.Int32
	srv := sequenceServer(t, &hits, frameStart+frameOverld, frameStart+frameBlock+frameText+frameStopped)
	evs, err := collectStream(t, replayCfg(srv.URL, 2))
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if hits.Load() != 2 || count(evs, SEMessageStart) != 1 || count(evs, SETextDelta) != 1 || count(evs, SEMessageStop) != 1 {
		t.Fatalf("hits=%d events=%+v", hits.Load(), evs)
	}
}

// After output was delivered a failure is surfaced, never replayed.
func TestNoReplayAfterCommit(t *testing.T) {
	var hits atomic.Int32
	srv := sequenceServer(t, &hits, frameStart+frameBlock+frameText+frameOverld)
	evs, err := collectStream(t, replayCfg(srv.URL, 3))
	var se *StreamError
	if !errors.As(err, &se) || hits.Load() != 1 || count(evs, SETextDelta) != 1 {
		t.Fatalf("err=%v hits=%d events=%+v", err, hits.Load(), evs)
	}
}

// Replays are bounded; the last error is surfaced.
func TestReplayBounded(t *testing.T) {
	var hits atomic.Int32
	srv := sequenceServer(t, &hits, frameStart+frameOverld)
	_, err := collectStream(t, replayCfg(srv.URL, 2))
	if err == nil || hits.Load() != 3 {
		t.Fatalf("err=%v hits=%d, want error after 3 attempts", err, hits.Load())
	}
}

// Off by default: one attempt, the error passes straight through.
func TestReplayOffByDefault(t *testing.T) {
	var hits atomic.Int32
	srv := sequenceServer(t, &hits, frameStart+frameOverld, frameStart+frameBlock+frameText+frameStopped)
	if _, err := collectStream(t, replayCfg(srv.URL, 0)); err == nil || hits.Load() != 1 {
		t.Fatalf("err=%v hits=%d", err, hits.Load())
	}
}

// Establishment failures (*HTTPError) are doStream's to retry, not replayed again.
func TestReplaySkipsHTTPError(t *testing.T) {
	var hits atomic.Int32
	srv := statusServer(t, &hits, http.StatusBadRequest, nil, "bad")
	if _, err := collectStream(t, replayCfg(srv.URL, 3)); StatusCode(err) != 400 || hits.Load() != 1 {
		t.Fatalf("err=%v hits=%d", err, hits.Load())
	}
}

// The host filter vetoes replaying errors it handles itself.
func TestReplayFilterVetoes(t *testing.T) {
	var hits atomic.Int32
	srv := sequenceServer(t, &hits, frameStart+frameOverld, frameStart+frameBlock+frameText+frameStopped)
	cfg := replayCfg(srv.URL, 2)
	cfg.StreamReplayFilter = func(error) bool { return false }
	if _, err := collectStream(t, cfg); err == nil || hits.Load() != 1 {
		t.Fatalf("err=%v hits=%d, want the vetoed error after 1 attempt", err, hits.Load())
	}
}
