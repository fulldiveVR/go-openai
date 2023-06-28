package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	. "github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/internal/test"
	"github.com/sashabaranov/go-openai/internal/test/checks"
)

func TestCompletionsWrongModel(t *testing.T) {
	config := DefaultConfig("whatever")
	config.BaseURL = "http://localhost/v1"
	client := NewClientWithConfig(config)

	_, err := client.CreateCompletion(
		context.Background(),
		CompletionRequest{
			MaxTokens: 5,
			Model:     GPT3Dot5Turbo,
		},
	)
	if !errors.Is(err, ErrCompletionUnsupportedModel) {
		t.Fatalf("CreateCompletion should return ErrCompletionUnsupportedModel, but returned: %v", err)
	}
}

func TestCompletionWithStream(t *testing.T) {
	config := DefaultConfig("whatever")
	client := NewClientWithConfig(config)

	ctx := context.Background()
	req := CompletionRequest{Stream: true}
	_, err := client.CreateCompletion(ctx, req)
	if !errors.Is(err, ErrCompletionStreamNotSupported) {
		t.Fatalf("CreateCompletion didn't return ErrCompletionStreamNotSupported")
	}
}

// TestCompletions Tests the completions endpoint of the API using the mocked server.
func TestCompletions(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/completions", handleCompletionEndpoint)
	req := CompletionRequest{
		MaxTokens: 5,
		Model:     "ada",
		Prompt:    "Lorem ipsum",
	}
	_, err := client.CreateCompletion(context.Background(), req)
	checks.NoError(t, err, "CreateCompletion error")
}

func TestCompletionsRateLimit(t *testing.T) {
	server := test.NewTestServer()
	server.RegisterHandler("/v1/completions", handleCompletionEndpoint)
	// create the test server
	var err error
	ts := server.OpenAITestServer()
	ts.Start()
	defer ts.Close()

	config := DefaultConfig(test.GetTestToken())
	config.EnableRateLimiter = true
	config.BaseURL = ts.URL + "/v1"
	client := NewClientWithConfig(config)
	ctx, cancel := context.WithCancel(context.Background())
	req := CompletionRequest{
		MaxTokens: 5,
		Model:     "ada",
	}
	req.Prompt = "Lorem ipsum"
	cancel()
	_, err = client.CreateCompletion(ctx, req)
	checks.ErrorContains(t, err, "context canceled", "CreateCompletion error")
}

// handleCompletionEndpoint Handles the completion endpoint by the test server.
func handleCompletionEndpoint(w http.ResponseWriter, r *http.Request) {
	var err error
	var resBytes []byte

	// completions only accepts POST requests
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
	var completionReq CompletionRequest
	if completionReq, err = getCompletionBody(r); err != nil {
		http.Error(w, "could not read request", http.StatusInternalServerError)
		return
	}
	res := CompletionResponse{
		ID:      strconv.Itoa(int(time.Now().Unix())),
		Object:  "test-object",
		Created: time.Now().Unix(),
		// would be nice to validate Model during testing, but
		// this may not be possible with how much upkeep
		// would be required / wouldn't make much sense
		Model: completionReq.Model,
	}
	// create completions
	n := completionReq.N
	if n == 0 {
		n = 1
	}
	for i := 0; i < n; i++ {
		// generate a random string of length completionReq.Length
		completionStr := strings.Repeat("a", completionReq.MaxTokens)
		if completionReq.Echo {
			completionStr = completionReq.Prompt.(string) + completionStr
		}
		res.Choices = append(res.Choices, CompletionChoice{
			Text:  completionStr,
			Index: i,
		})
	}
	inputTokens := numTokens(completionReq.Prompt.(string)) * n
	completionTokens := completionReq.MaxTokens * n
	res.Usage = Usage{
		PromptTokens:     inputTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      inputTokens + completionTokens,
	}
	resBytes, _ = json.Marshal(res)
	fmt.Fprintln(w, string(resBytes))
}

// getCompletionBody Returns the body of the request to create a completion.
func getCompletionBody(r *http.Request) (CompletionRequest, error) {
	completion := CompletionRequest{}
	// read the request body
	reqBody, err := io.ReadAll(r.Body)
	if err != nil {
		return CompletionRequest{}, err
	}
	err = json.Unmarshal(reqBody, &completion)
	if err != nil {
		return CompletionRequest{}, err
	}
	return completion, nil
}

func TestCompletionRequest_Tokens(t *testing.T) {
	testcases := []struct {
		name       string
		model      string
		prompt     any
		wantErr    error
		wantTokens int
	}{
		{
			name:    "test unknown model",
			model:   "unknown",
			prompt:  "Hello, world!",
			wantErr: errors.New("failed to tokenize prompt: model not supported: model not supported"),
		},
		{
			name:       "test1",
			model:      GPT3Dot5Turbo,
			prompt:     "Hello, world!",
			wantTokens: 4,
		},
		{
			name:       "test any prompt",
			model:      GPT3Dot5Turbo,
			prompt:     1,
			wantTokens: 0,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(tt *testing.T) {
			req := CompletionRequest{
				Model:  testcase.model,
				Prompt: testcase.prompt,
			}
			tokens, err := req.Tokens()
			if err != nil && testcase.wantErr == nil {
				tt.Fatalf("Tokens() returned unexpected error: %v", err)
			}

			if err != nil && testcase.wantErr != nil && err.Error() != testcase.wantErr.Error() {
				tt.Fatalf("Tokens() returned unexpected error: %v, want: %v", err, testcase.wantErr)
			}

			if tokens != testcase.wantTokens {
				tt.Fatalf("Tokens() returned unexpected number of tokens: %d, want: %d", tokens, testcase.wantTokens)
			}
		})
	}
}
