package openai

import (
	"context"
	"fmt"
	"sync"

	"golang.org/x/time/rate"
)

// Document Reference:
// https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quotas-limits#quotas-and-limits-reference
// https://platform.openai.com/docs/guides/rate-limits/overview

const (
	SecondsPerMinute                  = 60
	AzureDavinciRequestLimitPerMinute = 120
	AzureChatGPTRequestLimitPerMinute = 300
	AzureGPT4RequestLimitPerMinute    = 18
	AzureDefaultRequestLimitPerMinute = 300

	AzureDavinciTokensLimitPerMinute = 40000
	AzureChatGPTTokensLimitPerMinute = 120000
	AzureGPT4TokensLimitPerMinute    = 10000
	AzureGPT432kTokensLimitPerMinute = 32000
	AzureDefaultTokensLimitPerMinute = 120000

	OpenAITextAndEmbeddingRequestLimitPerMinute = 3500
	OpenAIChatRequestLimitPerMinute             = 3500
	OpenAIGPT4RequestLimitPerMinute             = 200
	OpenAIGPT432kRequestLimitPerMinute          = 20
	OpenAIAudioRequestLimitPerMinute            = 50
	OpenAIGPT3Dot5Turbo16kRequestLimitPerMinute = 2000
	OpenAIDefaultRequestLimitPerMinute          = 3500
	OpenAIGPT4TurboRequestLimitPerMinute        = 500

	OpenAIDavinciTokensLimitPerMinute          = 350000
	OpenAIAdaTokensLimitPerMinute              = 350000 * 200
	OpenAIChatTokensLimitPerMinute             = 90000
	OpenAIGPT3Dot5Turbo16kTokensLimitPerMinute = 180_000
	OpenAIGPT4TokensLimitPerMinute             = 10000
	OpenAIGPT432kTokensLimitPerMinute          = 150000
	OpenAIDefaultTokensLimitPerMinute          = 350000
	OpenAIGPT4TurboTokensLimitPerMinute        = 150000
)

type TokenCountable interface {
	Tokens() (int, error)
}

type RateLimiter interface {
	Wait(ctx context.Context, model string, tokens int) error
	WaitForRequest(ctx context.Context, model string, req TokenCountable) error
}

// MemRateLimiter is a token bucket based rate limiter for OpenAI API.
type MemRateLimiter struct {
	mutex           sync.RWMutex
	RequestLimiters map[string]*rate.Limiter
	TokensLimiters  map[string]*rate.Limiter
	apiType         APIType
}

func NewMemRateLimiter(apiType APIType) *MemRateLimiter {
	r := &MemRateLimiter{
		mutex:   sync.RWMutex{},
		apiType: apiType,
	}

	if r.apiType == APITypeAzure || r.apiType == APITypeAzureAD {
		r.RequestLimiters = r.newAzureRequestLimiters()
		r.TokensLimiters = r.newAzureTokensLimiters()
	}

	if r.apiType == APITypeOpenAI {
		r.RequestLimiters = r.newOpenAIRequestLimiters()
		r.TokensLimiters = r.newOpenAITokensLimiters()
	}

	return r
}

func (r *MemRateLimiter) WaitForRequest(ctx context.Context, model string, req TokenCountable) (err error) {
	if ctx == nil {
		return fmt.Errorf("context is nil")
	}

	if req == nil {
		return fmt.Errorf("request is nil")
	}

	var tokens int
	tokens, err = req.Tokens()
	if err != nil {
		err = fmt.Errorf("failed to get tokens count: %w", err)
		return
	}

	err = r.Wait(ctx, model, tokens)
	if err != nil {
		err = fmt.Errorf("failed to wait for rate limiter: %w", err)
		return
	}

	return
}

func (r *MemRateLimiter) Wait(ctx context.Context, model string, tokens int) (err error) {
	err = r.requestWait(ctx, model)
	if err != nil {
		return err
	}

	if tokens == 0 {
		return nil
	}

	return r.tokensWait(ctx, model, tokens)
}

func (r *MemRateLimiter) requestWait(ctx context.Context, model string) (err error) {
	var limiter *rate.Limiter
	var ok bool

	if r.RequestLimiters == nil {
		return nil
	}

	r.mutex.RLock()
	limiter, ok = r.RequestLimiters[model]
	r.mutex.RUnlock()
	if !ok {
		limiter = r.newLimiter(AzureDefaultRequestLimitPerMinute)
		if r.apiType == APITypeOpenAI {
			limiter = r.newLimiter(OpenAIDefaultRequestLimitPerMinute)
		}

		r.mutex.Lock()
		r.RequestLimiters[model] = limiter
		r.mutex.Unlock()
	}

	// if limiter is nil, it means that the model is not rate limited
	if limiter == nil {
		return nil
	}

	return limiter.WaitN(ctx, 1)
}

func (r *MemRateLimiter) tokensWait(ctx context.Context, model string, tokens int) (err error) {
	var limiter *rate.Limiter
	var ok bool

	if r.TokensLimiters == nil {
		return nil
	}

	r.mutex.RLock()
	limiter, ok = r.TokensLimiters[model]
	r.mutex.RUnlock()
	if !ok {
		limiter = r.newLimiter(AzureDefaultTokensLimitPerMinute)
		if r.apiType == APITypeOpenAI {
			limiter = r.newLimiter(OpenAIDefaultTokensLimitPerMinute)
		}

		r.mutex.Lock()
		r.TokensLimiters[model] = limiter
		r.mutex.Unlock()
	}

	// if limiter is nil, it means that the model is not rate limited
	if limiter == nil {
		return nil
	}

	return limiter.WaitN(ctx, tokens)
}

func (r *MemRateLimiter) newLimiter(minuteRate int) *rate.Limiter {
	return rate.NewLimiter(rate.Limit(minuteRate/SecondsPerMinute), minuteRate)
}

// newAzureRequestLimiters creates a map of request limiters for each azure openai model.
func (r *MemRateLimiter) newAzureRequestLimiters() map[string]*rate.Limiter {
	// The limiters that are not defined here are controlled by the DefaultRequestLimit.
	return map[string]*rate.Limiter{
		GPT3Davinci:       r.newLimiter(AzureDavinciRequestLimitPerMinute),
		GPT3Dot5Turbo:     r.newLimiter(AzureChatGPTRequestLimitPerMinute),
		GPT3Dot5Turbo0301: r.newLimiter(AzureChatGPTRequestLimitPerMinute),
		GPT4:              r.newLimiter(AzureGPT4RequestLimitPerMinute),
		GPT432K:           r.newLimiter(AzureGPT432kTokensLimitPerMinute),
	}
}

// newAzureTokensLimiters creates a map of tokens limiters for each azure openai model.
func (r *MemRateLimiter) newAzureTokensLimiters() map[string]*rate.Limiter {
	// The limiters that are not defined here are controlled by the DefaultTokensLimit.
	return map[string]*rate.Limiter{
		GPT3Davinci:       r.newLimiter(AzureDavinciTokensLimitPerMinute),
		GPT3Dot5Turbo:     r.newLimiter(AzureChatGPTTokensLimitPerMinute),
		GPT3Dot5Turbo0301: r.newLimiter(AzureChatGPTTokensLimitPerMinute),
		GPT4:              r.newLimiter(AzureGPT4TokensLimitPerMinute),
		GPT432K:           r.newLimiter(AzureGPT432kTokensLimitPerMinute),
	}
}

// newOpenAIRequestLimiters creates a map of request limiters for each openai model.
func (r *MemRateLimiter) newOpenAIRequestLimiters() map[string]*rate.Limiter {
	// The limiters that are not defined here are controlled by the DefaultRequestLimit.
	return map[string]*rate.Limiter{
		GPT3Davinci:          r.newLimiter(OpenAITextAndEmbeddingRequestLimitPerMinute),
		GPT3Dot5Turbo:        r.newLimiter(OpenAIChatRequestLimitPerMinute),
		GPT3Dot5Turbo0301:    r.newLimiter(OpenAIChatRequestLimitPerMinute),
		GPT3Dot5Turbo0613:    r.newLimiter(OpenAIChatRequestLimitPerMinute),
		GPT3Dot5Turbo16K:     r.newLimiter(OpenAIGPT3Dot5Turbo16kRequestLimitPerMinute),
		GPT3Dot5Turbo16K0613: r.newLimiter(OpenAIGPT3Dot5Turbo16kRequestLimitPerMinute),
		GPT4:                 r.newLimiter(OpenAIGPT4RequestLimitPerMinute),
		GPT40314:             r.newLimiter(OpenAIGPT4RequestLimitPerMinute),
		GPT432K:              r.newLimiter(OpenAIGPT432kRequestLimitPerMinute),
		GPT432K0314:          r.newLimiter(OpenAIGPT432kRequestLimitPerMinute),
		GPT4TurboPreview:     r.newLimiter(OpenAIGPT4TurboRequestLimitPerMinute),
	}
}

// newOpenAITokensLimiters creates a map of tokens limiters for each openai model.
func (r *MemRateLimiter) newOpenAITokensLimiters() map[string]*rate.Limiter {
	// The limiters that are not defined here are controlled by the DefaultTokensLimit.
	return map[string]*rate.Limiter{
		GPT3Davinci:            r.newLimiter(OpenAIDavinciTokensLimitPerMinute),
		GPT3TextAdaEmbeddingV2: r.newLimiter(OpenAIAdaTokensLimitPerMinute),
		GPT3Dot5Turbo:          r.newLimiter(OpenAIChatTokensLimitPerMinute),
		GPT3Dot5Turbo0301:      r.newLimiter(OpenAIChatTokensLimitPerMinute),
		GPT3Dot5Turbo0613:      r.newLimiter(OpenAIChatRequestLimitPerMinute),
		GPT3Dot5Turbo16K:       r.newLimiter(OpenAIGPT3Dot5Turbo16kTokensLimitPerMinute),
		GPT3Dot5Turbo16K0613:   r.newLimiter(OpenAIGPT3Dot5Turbo16kTokensLimitPerMinute),
		GPT4:                   r.newLimiter(OpenAIGPT4TokensLimitPerMinute),
		GPT432K:                r.newLimiter(OpenAIGPT432kTokensLimitPerMinute),
		GPT4TurboPreview:       r.newLimiter(OpenAIGPT4TurboTokensLimitPerMinute),
	}
}
