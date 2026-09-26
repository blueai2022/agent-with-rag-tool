// Package llmclient is a minimal OpenAI-compatible chat-completions client
// with tool (function) calling — the one HTTP call this whole repo needs.
// It works against OpenAI itself, or any OpenAI-compatible endpoint that
// supports tool calling (e.g. a local Ollama server's /v1 endpoint with a
// tool-calling capable model).
package llmclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// Message is one chat turn. Content is used for system/user/assistant text
// and for tool results (Role "tool", with ToolCallID set); ToolCalls is set
// on an assistant message that decided to call one or more tools instead of
// answering directly.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolCall is one function invocation requested by the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall names the requested tool and its arguments (a JSON object
// encoded as a string, per the OpenAI tool-calling wire format).
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool advertises one callable function to the model.
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function is a JSON-schema description of a callable tool.
type Function struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// ToolCallHandler is a function that handles a tool call and returns its output as a string.
type ToolCallHandler func(call ToolCall) string

// Client calls an OpenAI-compatible /chat/completions endpoint.
type Client struct {
	BaseURL string // e.g. "https://api.openai.com/v1" or "http://localhost:11434/v1"
	APIKey  string
	Model   string
	HTTP    *http.Client
}

// New constructs a Client. If httpClient is nil, http.DefaultClient is used.
func New(baseURL, apiKey, model string, httpClient *http.Client) *Client {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &Client{BaseURL: baseURL, APIKey: apiKey, Model: model, HTTP: httpClient}
}

type chatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// Chat sends the conversation so far plus the available tools, and returns
// the model's next message (either a final answer, or an assistant message
// carrying ToolCalls for the caller to execute).
func (c *Client) Chat(ctx context.Context, messages []Message, tools []Tool) (Message, error) {
	body, err := json.Marshal(chatRequest{Model: c.Model, Messages: messages, Tools: tools})
	if err != nil {
		return Message{}, fmt.Errorf("llmclient: encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Message{}, fmt.Errorf("llmclient: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return Message{}, fmt.Errorf("llmclient: request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return Message{}, fmt.Errorf("llmclient: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return Message{}, fmt.Errorf("llmclient: %s returned %d: %s", c.BaseURL, resp.StatusCode, respBody)
	}

	var out chatResponse
	if err := json.Unmarshal(respBody, &out); err != nil {
		return Message{}, fmt.Errorf("llmclient: decode response: %w", err)
	}
	if out.Error != nil {
		return Message{}, fmt.Errorf("llmclient: api error: %s", out.Error.Message)
	}
	if len(out.Choices) == 0 {
		return Message{}, fmt.Errorf("llmclient: response had no choices")
	}
	return out.Choices[0].Message, nil
}
