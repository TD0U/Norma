package llm

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// HTTPError is a non-200 response from the model endpoint, returned (possibly
// wrapped) by Stream and Complete once establishment retries are exhausted or
// the status is not retryable. Use errors.As to inspect it. The Error() text is
// unchanged from earlier versions ("<format>: status <code>: <body>"), so hosts
// that matched on the string keep working.
type HTTPError struct {
	Format     string        // "anthropic" | "openai" | "openai-responses"
	StatusCode int           // HTTP status
	Body       string        // response body, trimmed
	Header     http.Header   // response headers
	RetryAfter time.Duration // parsed Retry-After; 0 when absent or invalid
	// Class is the ErrorClassifier's verdict, or ClassDefault when none ran.
	Class ErrorClass
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("%s: status %d: %s", e.Format, e.StatusCode, e.Body)
}

// ErrorClass is a host's classification of a failed response.
type ErrorClass int

const (
	// ClassDefault keeps the built-in decision (retry 408/429/5xx).
	ClassDefault ErrorClass = iota
	// ClassRetryable forces a retry (still bounded by MaxRetries).
	ClassRetryable
	// ClassFatal stops retrying immediately.
	ClassFatal
	// ClassQuotaExhausted is ClassFatal with a reason: the account's quota,
	// balance or billing is exhausted, so retrying the same provider cannot
	// help. Hosts typically fail over on it. Test with IsQuotaExhausted.
	ClassQuotaExhausted
)

// ErrorClassifier inspects a failed response before the retry decision.
// Typical use: tell an exhausted-quota 429 apart from an ordinary rate limit —
// both are 429, only the body differs, and retrying the former burns the whole
// backoff ladder. Return ClassDefault to keep the built-in decision.
type ErrorClassifier func(e *HTTPError) ErrorClass

// StreamError is an error frame received inside an otherwise successful (200)
// stream, e.g. Anthropic's overloaded_error. Type is the provider's error type
// when it sends one ("overloaded_error", "rate_limit_error", …).
type StreamError struct {
	Format  string
	Type    string
	Message string
}

func (e *StreamError) Error() string { return e.Format + ": " + e.Message }

// IsQuotaExhausted reports whether err is an HTTPError the classifier marked
// ClassQuotaExhausted.
func IsQuotaExhausted(err error) bool {
	var he *HTTPError
	return errors.As(err, &he) && he.Class == ClassQuotaExhausted
}

// StatusCode returns the HTTP status carried by err, or 0 when err is not an
// HTTPError.
func StatusCode(err error) int {
	var he *HTTPError
	if errors.As(err, &he) {
		return he.StatusCode
	}
	return 0
}

// maxRetryAfter caps an honoured Retry-After, so a bogus or hostile value
// (hours, years) cannot stall a request; past the cap the normal backoff is used.
const maxRetryAfter = 60 * time.Second

// parseRetryAfter reads a Retry-After header: delay-seconds or an HTTP date.
// Returns 0 when absent, invalid, in the past, or above maxRetryAfter.
func parseRetryAfter(h http.Header, now time.Time) time.Duration {
	v := strings.TrimSpace(h.Get("Retry-After"))
	if v == "" {
		return 0
	}
	var d time.Duration
	if secs, err := strconv.Atoi(v); err == nil {
		d = time.Duration(secs) * time.Second
	} else if t, err := http.ParseTime(v); err == nil {
		d = t.Sub(now)
	}
	if d <= 0 || d > maxRetryAfter {
		return 0
	}
	return d
}
