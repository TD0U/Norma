package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func streamErr(t *testing.T, cfg Config) error {
	t.Helper()
	p, err := NewProvider(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range p.Stream(context.Background(), CompletionRequest{Messages: []Message{UserText("x")}}) {
		if e != nil {
			return e
		}
	}
	return nil
}

func statusServer(t *testing.T, hits *atomic.Int32, status int, header map[string]string, body string) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		for k, v := range header {
			w.Header().Set(k, v)
		}
		w.WriteHeader(status)
		fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// A final non-200 is a typed *HTTPError whose text is unchanged.
func TestHTTPErrorTyped(t *testing.T) {
	var hits atomic.Int32
	srv := statusServer(t, &hits, http.StatusBadRequest, nil, " bad input ")
	err := streamErr(t, Config{Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	var he *HTTPError
	if !errors.As(err, &he) || he.StatusCode != 400 || he.Body != "bad input" {
		t.Fatalf("err = %#v", err)
	}
	if err.Error() != "anthropic: status 400: bad input" || StatusCode(err) != 400 {
		t.Fatalf("text = %q", err.Error())
	}
	if hits.Load() != 1 {
		t.Fatalf("400 retried: %d hits", hits.Load())
	}
}

// The classifier can stop a 429 from being retried and label it quota.
func TestErrorClassifierQuota(t *testing.T) {
	var hits atomic.Int32
	srv := statusServer(t, &hits, http.StatusTooManyRequests, nil, `{"error":{"code":"insufficient_quota"}}`)
	cfg := Config{Format: FormatOpenAI, BaseURL: srv.URL, APIKey: "k", Model: "m", MaxRetries: 3, RetryInterval: time.Millisecond,
		ErrorClassifier: func(e *HTTPError) ErrorClass {
			if strings.Contains(e.Body, "insufficient_quota") {
				return ClassQuotaExhausted
			}
			return ClassDefault
		}}
	err := streamErr(t, cfg)
	if !IsQuotaExhausted(err) {
		t.Fatalf("err = %v, want quota exhausted", err)
	}
	if hits.Load() != 1 {
		t.Fatalf("quota 429 retried: %d hits", hits.Load())
	}
	// Without the classifier the same 429 is retried (built-in rule).
	hits.Store(0)
	cfg.ErrorClassifier = nil
	if err := streamErr(t, cfg); IsQuotaExhausted(err) || hits.Load() != 4 {
		t.Fatalf("default: err=%v hits=%d, want plain error after 4 hits", err, hits.Load())
	}
}

// Retry-After longer than the backoff ladder is honoured.
func TestRetryAfterHonoured(t *testing.T) {
	var hits atomic.Int32
	srv := statusServer(t, &hits, http.StatusServiceUnavailable, map[string]string{"Retry-After": "1"}, "busy")
	start := time.Now()
	err := streamErr(t, Config{Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Model: "m", MaxRetries: 1, RetryInterval: time.Millisecond})
	var he *HTTPError
	if !errors.As(err, &he) || he.RetryAfter != time.Second {
		t.Fatalf("err = %v", err)
	}
	if took := time.Since(start); took < 900*time.Millisecond {
		t.Fatalf("Retry-After ignored: retried after %s", took)
	}
}

func TestParseRetryAfter(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	h := func(v string) http.Header { return http.Header{"Retry-After": {v}} }
	cases := map[string]time.Duration{
		"5": 5 * time.Second, "": 0, "abc": 0, "-3": 0, "3600": 0, // over cap → ignored
		now.Add(10 * time.Second).Format(http.TimeFormat): 10 * time.Second,
		now.Add(-time.Minute).Format(http.TimeFormat):     0,
	}
	for v, want := range cases {
		if got := parseRetryAfter(h(v), now); got != want {
			t.Errorf("Retry-After %q = %s, want %s", v, got, want)
		}
	}
}

// An error frame inside a 200 stream is a typed *StreamError.
func TestStreamErrorFrameTyped(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("content-type", "text/event-stream")
		fmt.Fprint(w, "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n")
	}))
	defer srv.Close()
	err := streamErr(t, Config{Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Model: "m"})
	var se *StreamError
	if !errors.As(err, &se) || se.Type != "overloaded_error" || err.Error() != "anthropic: Overloaded" {
		t.Fatalf("err = %#v", err)
	}
}
