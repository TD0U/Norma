package llm

import (
	"errors"
	"fmt"
	"io"
	"sync"
	"time"
)

// Timeouts for one model request. Both are off by default (zero) so existing
// hosts keep their behaviour; a long-running agent should set both.
//
//   - Config.ResponseHeaderTimeout bounds the wait between sending the request
//     and receiving response headers. It does NOT bound the streamed body, so a
//     slow but live reasoning stream is never cut. A trip is a transport error
//     before any stream data, which doStream's establishment retries cover.
//
//   - Config.StreamIdleTimeout bounds the gap between two reads of the
//     response body once streaming has started. A model that keeps emitting —
//     even slowly, even only keep-alive pings — never trips it; a stream that
//     goes silent mid-response does. The body is closed, the blocked read
//     returns, and the stream ends with ErrStreamIdle. MaxDuration cannot catch
//     this case: it only bounds tool execution, never the model stream.

// ErrStreamIdle is returned (wrapped) when a streaming response produces no
// bytes for longer than Config.StreamIdleTimeout. Test with errors.Is. It is a
// transport failure, so hosts may treat it as retryable; whether a retry is
// safe depends on whether output was already handed to the consumer.
var ErrStreamIdle = errors.New("llm: stream idle timeout")

// idleTimeoutBody wraps a response body and closes it when no Read completes
// within d. Close is safe to call concurrently with a blocked Read on
// net/http bodies, which is what unblocks the stalled read.
type idleTimeoutBody struct {
	body  io.ReadCloser
	d     time.Duration
	timer *time.Timer

	mu      sync.Mutex
	tripped bool
	closed  bool
}

func newIdleTimeoutBody(body io.ReadCloser, d time.Duration) *idleTimeoutBody {
	b := &idleTimeoutBody{body: body, d: d}
	b.timer = time.AfterFunc(d, b.trip)
	return b
}

func (b *idleTimeoutBody) trip() {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return
	}
	b.tripped = true
	b.mu.Unlock()
	_ = b.body.Close()
}

func (b *idleTimeoutBody) Read(p []byte) (int, error) {
	n, err := b.body.Read(p)
	b.mu.Lock()
	tripped := b.tripped
	b.mu.Unlock()
	if tripped {
		return n, fmt.Errorf("%w: no data for %s", ErrStreamIdle, b.d)
	}
	if n > 0 {
		b.timer.Reset(b.d)
	}
	return n, err
}

func (b *idleTimeoutBody) Close() error {
	b.mu.Lock()
	b.closed = true
	b.mu.Unlock()
	b.timer.Stop()
	return b.body.Close()
}
