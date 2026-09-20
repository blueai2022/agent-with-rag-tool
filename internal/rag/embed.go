package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
)

// Embedder returns one embedding vector per input string, in the same order.
type Embedder interface {
	Embed(ctx context.Context, inputs []string) ([][]float32, error)
}

// OllamaEmbedder calls an OpenAI-compatible /v1/embeddings endpoint (Ollama by
// default at http://localhost:11434).
type OllamaEmbedder struct {
	baseURL string
	model   string
	apiKey  string
	client  *http.Client
}

// NewOllamaEmbedder builds an embedder for the given base URL, model, and
// optional bearer token. An empty base URL defaults to localhost Ollama.
func NewOllamaEmbedder(baseURL, model, apiKey string) *OllamaEmbedder {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	return &OllamaEmbedder{
		baseURL: strings.TrimRight(baseURL, "/"),
		model:   model,
		apiKey:  apiKey,
		client:  &http.Client{},
	}
}

type embedRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model"`
}

type embedData struct {
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type embedResponse struct {
	Data []embedData `json:"data"`
}

func (e *OllamaEmbedder) Embed(ctx context.Context, inputs []string) ([][]float32, error) {
	body, err := json.Marshal(embedRequest{Input: inputs, Model: e.model})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if e.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.apiKey)
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("rag: embeddings status %d", resp.StatusCode)
	}
	var out embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Data) != len(inputs) {
		return nil, fmt.Errorf("rag: embeddings returned %d vectors for %d inputs", len(out.Data), len(inputs))
	}
	// Order by index so the result matches input order (providers may reorder).
	sort.Slice(out.Data, func(i, j int) bool { return out.Data[i].Index < out.Data[j].Index })
	vectors := make([][]float32, len(inputs))
	for i := range out.Data {
		vectors[i] = out.Data[i].Embedding
	}
	return vectors, nil
}

// FakeEmbedder returns deterministic vectors for tests (no live model). When
// Func is set it is used verbatim; otherwise the input is hashed into a
// dim-length vector.
type FakeEmbedder struct {
	Dim  int
	Func func(input string) []float32
}

func (f *FakeEmbedder) Embed(ctx context.Context, inputs []string) ([][]float32, error) {
	out := make([][]float32, len(inputs))
	for i, in := range inputs {
		if f.Func != nil {
			out[i] = f.Func(in)
		} else {
			out[i] = fakeVec(in, f.Dim)
		}
	}
	return out, nil
}

func fakeVec(input string, dim int) []float32 {
	v := make([]float32, dim)
	if dim == 0 {
		return v
	}
	for i, b := range []byte(input) {
		v[i%dim] += float32(b) / 255.0
	}
	return v
}
