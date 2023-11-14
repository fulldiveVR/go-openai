package openai_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"testing"

	. "github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/internal/test/checks"
)

func TestEmbedding(t *testing.T) {
	embeddedModels := []EmbeddingModel{
		AdaSimilarity,
		BabbageSimilarity,
		CurieSimilarity,
		DavinciSimilarity,
		AdaSearchDocument,
		AdaSearchQuery,
		BabbageSearchDocument,
		BabbageSearchQuery,
		CurieSearchDocument,
		CurieSearchQuery,
		DavinciSearchDocument,
		DavinciSearchQuery,
		AdaCodeSearchCode,
		AdaCodeSearchText,
		BabbageCodeSearchCode,
		BabbageCodeSearchText,
	}
	for _, model := range embeddedModels {
		// test embedding request with strings (simple embedding request)
		embeddingReq := EmbeddingRequest{
			Input: []string{
				"The food was delicious and the waiter",
				"Other examples of embedding request",
			},
			Model: model,
		}
		// marshal embeddingReq to JSON and confirm that the model field equals
		// the AdaSearchQuery type
		marshaled, err := json.Marshal(embeddingReq)
		checks.NoError(t, err, "Could not marshal embedding request")
		if !bytes.Contains(marshaled, []byte(`"model":"`+model.String()+`"`)) {
			t.Fatalf("Expected embedding request to contain model field")
		}

		// test embedding request with strings
		embeddingReqStrings := EmbeddingRequestStrings{
			Input: []string{
				"The food was delicious and the waiter",
				"Other examples of embedding request",
			},
			Model: model,
		}
		marshaled, err = json.Marshal(embeddingReqStrings)
		checks.NoError(t, err, "Could not marshal embedding request")
		if !bytes.Contains(marshaled, []byte(`"model":"`+model.String()+`"`)) {
			t.Fatalf("Expected embedding request to contain model field")
		}

		// test embedding request with tokens
		embeddingReqTokens := EmbeddingRequestTokens{
			Input: [][]int{
				{464, 2057, 373, 12625, 290, 262, 46612},
				{6395, 6096, 286, 11525, 12083, 2581},
			},
			Model: model,
		}
		marshaled, err = json.Marshal(embeddingReqTokens)
		checks.NoError(t, err, "Could not marshal embedding request")
		if !bytes.Contains(marshaled, []byte(`"model":"`+model.String()+`"`)) {
			t.Fatalf("Expected embedding request to contain model field")
		}
	}
}

func TestEmbeddingModel(t *testing.T) {
	var em EmbeddingModel
	err := em.UnmarshalText([]byte("text-similarity-ada-001"))
	checks.NoError(t, err, "Could not marshal embedding model")

	if em != AdaSimilarity {
		t.Errorf("Model is not equal to AdaSimilarity")
	}

	err = em.UnmarshalText([]byte("some-non-existent-model"))
	checks.NoError(t, err, "Could not marshal embedding model")
	if em != Unknown {
		t.Errorf("Model is not equal to Unknown")
	}
}

// TODO: fix this test
func TestEmbeddingEndpoint(t *testing.T) {
	client, server, teardown := setupOpenAITestServer()
	defer teardown()
	server.RegisterHandler(
		"/v1/embeddings",
		func(w http.ResponseWriter, r *http.Request) {
			resBytes, _ := json.Marshal(EmbeddingResponse{})
			fmt.Fprintln(w, string(resBytes))
		},
	)
	// test create embeddings with strings (simple embedding request)
	_, err := client.CreateEmbeddings(context.Background(), EmbeddingRequest{Model: EmbeddingModel(1)})
	checks.NoError(t, err, "CreateEmbeddings error")

	// test create embeddings with strings
	_, err = client.CreateEmbeddings(context.Background(), EmbeddingRequestStrings{})
	checks.NoError(t, err, "CreateEmbeddings strings error")

	// test create embeddings with tokens
	_, err = client.CreateEmbeddings(context.Background(), EmbeddingRequestTokens{})
	checks.NoError(t, err, "CreateEmbeddings tokens error")
}

func TestEmbeddingRequest_Tokens(t *testing.T) {
	testcases := []struct {
		name       string
		model      EmbeddingModel
		input      []string
		wantErr    error
		wantTokens int
	}{
		{
			name:    "test unknown model",
			wantErr: errors.New("failed to tokenize prompt: model not supported: model not supported"),
		},
		{
			name:  "test1",
			model: AdaEmbeddingV2,
			input: []string{
				"The food was delicious and the waiter",
			},
			wantTokens: 7,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(tt *testing.T) {
			req := EmbeddingRequest{
				Model: testcase.model,
				Input: testcase.input,
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
