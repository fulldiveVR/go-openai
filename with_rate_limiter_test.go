package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"testing"

	. "github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/internal/test"
	"github.com/sashabaranov/go-openai/internal/test/checks"
	"github.com/sashabaranov/go-openai/jsonschema"
)

func TestCreateChatCompletionStreamRateLimit(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")

		// Send test responses
		dataBytes := []byte{}
		dataBytes = append(dataBytes, []byte("event: message\n")...)
		//nolint:lll
		data := `{"id":"1","object":"completion","created":1598069254,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"response1"},"finish_reason":"max_tokens"}]}`
		dataBytes = append(dataBytes, []byte("data: "+data+"\n\n")...)

		dataBytes = append(dataBytes, []byte("event: message\n")...)
		//nolint:lll
		data = `{"id":"2","object":"completion","created":1598069255,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"response2"},"finish_reason":"max_tokens"}]}`
		dataBytes = append(dataBytes, []byte("data: "+data+"\n\n")...)

		dataBytes = append(dataBytes, []byte("event: done\n")...)
		dataBytes = append(dataBytes, []byte("data: [DONE]\n\n")...)

		_, err := w.Write(dataBytes)
		checks.NoError(t, err, "Write error")
	})

	ctx, cancel := context.WithCancel(context.Background())

	cancel()
	_, err := client.CreateChatCompletionStream(ctx, ChatCompletionRequest{
		MaxTokens: 5,
		Model:     GPT3Dot5Turbo,
		Messages: []ChatCompletionMessage{
			{
				Role:    ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
		Stream: true,
	})
	checks.ErrorContains(t, err, "failed to wait for rate limiter: context canceled")
}

func TestCreateChatCompletionStreamRateLimitError(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(429)

		// Send test responses
		dataBytes := []byte(`{"error":{` +
			`"message": "You are sending requests too quickly.",` +
			`"type":"rate_limit_reached",` +
			`"param":null,` +
			`"code":"rate_limit_reached"}}`)

		_, err := w.Write(dataBytes)
		checks.NoError(t, err, "Write error")
	})
	_, err := client.CreateChatCompletionStream(context.Background(), ChatCompletionRequest{
		MaxTokens: 5,
		Model:     GPT3Dot5Turbo,
		Messages: []ChatCompletionMessage{
			{
				Role:    ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
		Stream: true,
	})
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Errorf("TestCreateChatCompletionStreamRateLimitError did not return APIError")
	}
	t.Logf("%+v\n", apiErr)
}

func TestAzureCreateChatCompletionStreamRateLimitError(t *testing.T) {
	wantCode := "429"
	wantMessage := "Requests to the Creates a completion for the chat message Operation under Azure OpenAI API " +
		"version 2023-03-15-preview have exceeded token rate limit of your current OpenAI S0 pricing tier. " +
		"Please retry after 20 seconds. " +
		"Please go here: https://aka.ms/oai/quotaincrease if you would like to further increase the default rate limit."

	client, server, teardown := setupAzureTestServer()
	defer teardown()
	server.RegisterHandler("/openai/deployments/gpt-35-turbo/chat/completions",
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			// Send test responses
			dataBytes := []byte(`{"error": { "code": "` + wantCode + `", "message": "` + wantMessage + `"}}`)
			_, err := w.Write(dataBytes)

			checks.NoError(t, err, "Write error")
		})

	apiErr := &APIError{}
	_, err := client.CreateChatCompletionStream(context.Background(), ChatCompletionRequest{
		MaxTokens: 5,
		Model:     GPT3Dot5Turbo,
		Messages: []ChatCompletionMessage{
			{
				Role:    ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
		Stream: true,
	})
	if !errors.As(err, &apiErr) {
		t.Errorf("Did not return APIError: %+v\n", apiErr)
		return
	}
	if apiErr.HTTPStatusCode != http.StatusTooManyRequests {
		t.Errorf("Did not return HTTPStatusCode got = %d, want = %d\n", apiErr.HTTPStatusCode, http.StatusTooManyRequests)
		return
	}
	code, ok := apiErr.Code.(string)
	if !ok || code != wantCode {
		t.Errorf("Did not return Code. got = %v, want = %s\n", apiErr.Code, wantCode)
		return
	}
	if apiErr.Message != wantMessage {
		t.Errorf("Did not return Message. got = %s, want = %s\n", apiErr.Message, wantMessage)
		return
	}
}

func TestChatCompletionsRateLimit(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/chat/completions", handleChatCompletionEndpoint)
	t.Run("bytes", func(t *testing.T) {
		//nolint:lll
		msg := json.RawMessage(`{"properties":{"count":{"type":"integer","description":"total number of words in sentence"},"words":{"items":{"type":"string"},"type":"array","description":"list of words in sentence"}},"type":"object","required":["count","words"]}`)
		_, err := client.CreateChatCompletion(context.Background(), ChatCompletionRequest{
			MaxTokens: 5,
			Model:     GPT3Dot5Turbo0613,
			Messages: []ChatCompletionMessage{
				{
					Role:    ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
			Functions: []FunctionDefinition{{
				Name:       "test",
				Parameters: &msg,
			}},
		})
		checks.NoError(t, err, "CreateChatCompletion with functions error")
	})
	t.Run("struct", func(t *testing.T) {
		type testMessage struct {
			Count int      `json:"count"`
			Words []string `json:"words"`
		}
		msg := testMessage{
			Count: 2,
			Words: []string{"hello", "world"},
		}
		_, err := client.CreateChatCompletion(context.Background(), ChatCompletionRequest{
			MaxTokens: 5,
			Model:     GPT3Dot5Turbo0613,
			Messages: []ChatCompletionMessage{
				{
					Role:    ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
			Functions: []FunctionDefinition{{
				Name:       "test",
				Parameters: &msg,
			}},
		})
		checks.NoError(t, err, "CreateChatCompletion with functions error")
	})
	t.Run("JSONSchemaDefinition", func(t *testing.T) {
		_, err := client.CreateChatCompletion(context.Background(), ChatCompletionRequest{
			MaxTokens: 5,
			Model:     GPT3Dot5Turbo0613,
			Messages: []ChatCompletionMessage{
				{
					Role:    ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
			Functions: []FunctionDefinition{{
				Name: "test",
				Parameters: &jsonschema.Definition{
					Type: jsonschema.Object,
					Properties: map[string]jsonschema.Definition{
						"count": {
							Type:        jsonschema.Number,
							Description: "total number of words in sentence",
						},
						"words": {
							Type:        jsonschema.Array,
							Description: "list of words in sentence",
							Items: &jsonschema.Definition{
								Type: jsonschema.String,
							},
						},
						"enumTest": {
							Type: jsonschema.String,
							Enum: []string{"hello", "world"},
						},
					},
				},
			}},
		})
		checks.NoError(t, err, "CreateChatCompletion with functions error")
	})
	t.Run("JSONSchemaDefinitionWithFunctionDefine", func(t *testing.T) {
		// this is a compatibility check
		_, err := client.CreateChatCompletion(context.Background(), ChatCompletionRequest{
			MaxTokens: 5,
			Model:     GPT3Dot5Turbo0613,
			Messages: []ChatCompletionMessage{
				{
					Role:    ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
			Functions: []FunctionDefine{{
				Name: "test",
				Parameters: &jsonschema.Definition{
					Type: jsonschema.Object,
					Properties: map[string]jsonschema.Definition{
						"count": {
							Type:        jsonschema.Number,
							Description: "total number of words in sentence",
						},
						"words": {
							Type:        jsonschema.Array,
							Description: "list of words in sentence",
							Items: &jsonschema.Definition{
								Type: jsonschema.String,
							},
						},
						"enumTest": {
							Type: jsonschema.String,
							Enum: []string{"hello", "world"},
						},
					},
				},
			}},
		})
		checks.NoError(t, err, "CreateChatCompletion with functions error")
	})
}

func TestChatCompletionsRateLimitErr(t *testing.T) {
	server := test.NewTestServer()
	server.RegisterHandler("/v1/chat/completions", handleChatCompletionEndpoint)
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
	req := ChatCompletionRequest{
		MaxTokens: 5,
		Model:     GPT3Dot5Turbo,
		Messages: []ChatCompletionMessage{
			{
				Role:    ChatMessageRoleUser,
				Content: "Hello!",
			},
		},
	}
	cancel()
	_, err = client.CreateChatCompletion(ctx, req)
	checks.HasError(t, err, "CreateChatCompletion error")
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

func TestEmbeddingRateLimit(t *testing.T) {
	server := test.NewTestServer()
	server.RegisterHandler(
		"/v1/embeddings",
		func(w http.ResponseWriter, r *http.Request) {
			resBytes, _ := json.Marshal(EmbeddingResponse{})
			fmt.Fprintln(w, string(resBytes))
		},
	)
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
	cancel()
	_, err = client.CreateEmbeddings(ctx, EmbeddingRequest{
		Model: AdaEmbeddingV2,
	})
	checks.ErrorContains(t, err, "context canceled", "CreateEmbeddings error")
}

func TestCreateCompletionStreamRateLimitError(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(429)

		// Send test responses
		dataBytes := []byte(`{"error":{` +
			`"message": "You are sending requests too quickly.",` +
			`"type":"rate_limit_reached",` +
			`"param":null,` +
			`"code":"rate_limit_reached"}}`)

		_, err := w.Write(dataBytes)
		checks.NoError(t, err, "Write error")
	})

	var apiErr *APIError
	_, err := client.CreateCompletionStream(context.Background(), CompletionRequest{
		MaxTokens: 5,
		Model:     GPT3Ada,
		Prompt:    "Hello!",
		Stream:    true,
	})
	if !errors.As(err, &apiErr) {
		t.Errorf("TestCreateCompletionStreamRateLimitError did not return APIError")
	}
	t.Logf("%+v\n", apiErr)
}
