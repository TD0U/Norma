package llm

import (
	"context"
	"errors"
	"iter"
	"strings"
	"time"
)

// replayProvider replays a stream that fails before committing any output.
// Events that carry no output (message_start, empty deltas, signatures) are
// held back until the first committing event, so a replay never shows the
// caller a second message_start or partial content.
type replayProvider struct {
	inner    Provider
	retries  int
	interval time.Duration
	filter   func(error) bool
}

func (r *replayProvider) Complete(ctx context.Context, req CompletionRequest) (Message, string, Usage, error) {
	return r.inner.Complete(ctx, req)
}

func (r *replayProvider) Stream(ctx context.Context, req CompletionRequest) iter.Seq2[StreamEvent, error] {
	return func(yield func(StreamEvent, error) bool) {
		for attempt := 0; ; attempt++ {
			var pending []StreamEvent
			committed := false
			var streamErr error
			for ev, err := range r.inner.Stream(ctx, req) {
				if err != nil {
					streamErr = err
					break
				}
				if !committed && !commitsOutput(ev) {
					pending = append(pending, ev)
					continue
				}
				if !committed {
					for _, p := range pending {
						if !yield(p, nil) {
							return
						}
					}
					pending, committed = nil, true
				}
				if !yield(ev, nil) {
					return
				}
			}
			if streamErr == nil || committed || attempt >= r.retries || !replayable(ctx, streamErr) || (r.filter != nil && !r.filter(streamErr)) {
				for _, p := range pending {
					if !yield(p, nil) {
						return
					}
				}
				if streamErr != nil {
					yield(StreamEvent{}, streamErr)
				}
				return
			}
			d := r.interval
			if d <= 0 {
				d = expBackoff(attempt)
			}
			if !backoffSleep(ctx, d) {
				yield(StreamEvent{}, ctx.Err())
				return
			}
		}
	}
}

// commitsOutput reports whether an event delivers output the caller may act
// on. Once one has been yielded, a failure can no longer be replayed.
func commitsOutput(ev StreamEvent) bool {
	switch ev.Type {
	case SETextDelta, SEThinkingDelta, SEToolInputJSON:
		return ev.Text != ""
	case SEToolUseStart, SEMessageDelta, SEMessageStop:
		return true
	}
	return false
}

// replayable: in-stream failures only. Cancellation is the caller's decision;
// an *HTTPError already went through doStream's establishment retries; an
// over-long request fails identically on replay.
func replayable(ctx context.Context, err error) bool {
	if ctx.Err() != nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var he *HTTPError
	if errors.As(err, &he) {
		return false
	}
	s := strings.ToLower(err.Error())
	for _, m := range []string{"too long", "context length", "context_length", "maximum context"} {
		if strings.Contains(s, m) {
			return false
		}
	}
	return true
}
