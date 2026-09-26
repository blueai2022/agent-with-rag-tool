// Package agent is the tool-calling loop itself: the part of this repo that
// actually differs in shape from the production pipeline. Instead of the
// pipeline pre-building a fixed candidate lineup and making one LLM call
// (internal/overwatch.SelectCode in the production repo), the model here
// decides for itself when and how to call search_icd10, can call it more
// than once, and only then emits the final JSON answer.
package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"icd10-agent/internal/llmclient"
	"icd10-agent/internal/rag"
)

// maxSteps bounds the tool-call loop so a model that never converges on a
// final answer fails loudly.
const maxSteps = 6

// Result is the agent's final decision, matching the production selector's
// {final_code, reason} contract (prompts/icd_select.txt in the source repo).
type Result struct {
	FinalCode string `json:"final_code"`
	Reason    string `json:"reason,omitempty"`
}

// Agent wires an LLM client, a search tool over a corpus, and a system
// prompt into a single-purpose ReAct-style loop.
type Agent struct {
	llm          *llmclient.Client
	retriever    *rag.Retriever
	systemPrompt string
}

// New constructs an Agent.
func New(llm *llmclient.Client, r *rag.Retriever, systemPrompt string) *Agent {
	return &Agent{llm: llm, retriever: r, systemPrompt: systemPrompt}
}

// Select runs the agent loop for one (quotedText, selectedICD) case and
// returns the final decision along with the full message transcript (useful
// for --verbose / debugging, and for showing the agent's tool-call trail).
func (a *Agent) Select(
	ctx context.Context,
	quotedText, selectedICD string,
) (Result, []llmclient.Message, error) {
	tool, dispatch := searchICD10Tool(a.retriever)
	tools := []llmclient.Tool{tool}

	messages := []llmclient.Message{
		{Role: "system", Content: a.systemPrompt},
		{Role: "user", Content: fmt.Sprintf("Quoted evidence: %s\nUpstream code: %s", quotedText, selectedICD)},
	}

	for range maxSteps {
		rsp, err := a.llm.Chat(ctx, messages, tools)
		if err != nil {
			return Result{}, messages, fmt.Errorf("agent: chat call failed: %w", err)
		}

		// Append the model response, including any tool calls.
		messages = append(messages, rsp)

		if len(rsp.ToolCalls) > 0 {
			for _, call := range rsp.ToolCalls {
				out := dispatch(call)
				messages = append(messages, llmclient.Message{ // Append a tool response
					Role:       "tool",
					ToolCallID: call.ID,
					Content:    out,
				})
			}

			continue
		}

		// Final model answer parsing and validation
		result, err := parseResult(rsp.Content)

		if err != nil && !isBadJSON(err) { // true fatal errors
			return Result{}, messages, fmt.Errorf("agent: parse final answer: %w", err)
		}

		if isBadJSON(err) { // handle small LLMs that may produce malformed JSON
			messages = append(messages, llmclient.Message{
				Role: "user",
				Content: fmt.Sprintf(
					"Your previous reply was not valid JSON, so it was discarded.\n"+
						"Quoted evidence: %s\nUpstream code: %s\n"+
						"Respond now with ONLY a JSON object matching the schema: "+
						`{"final_code": "string", "reason": "string"}. Do not call any tools.`,
					quotedText, selectedICD),
			})

			continue // retry
		}

		if result.FinalCode == "" { // handle small LLMs that may select nothing
			messages = append(messages, llmclient.Message{
				Role: "user",
				Content: fmt.Sprintf(
					"You returned an empty final_code.\nQuoted evidence: %s\nUpstream code: %s\n"+
						"Re-invoke the search_icd10 tool and review the results in this conversation — if "+
						"one of them fits the evidence, respond with a JSON object using that "+
						"code. Only return an empty final_code if none fit.",
					quotedText, selectedICD),
			})

			continue // retry
		}

		return result, messages, nil
	}

	return Result{}, messages, fmt.Errorf("agent: exceeded %d steps without a final answer", maxSteps)
}

// isBadJSON reports whether parseResult failed to decode the model's reply,
// gating the malformed-JSON retry point in Select. This covers both clearly
// invalid syntax and a truncated reply (io.ErrUnexpectedEOF, e.g. the model
// was cut off mid-generation) — both are retryable, not fatal.
func isBadJSON(err error) bool {
	if err == nil {
		return false
	}

	if _, ok := errors.AsType[*json.SyntaxError](err); ok {
		return true
	}

	if errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}

	return false
}

// parseResult extracts the {"final_code","reason"} JSON object from the
// model's final message
func parseResult(content string) (Result, error) {
	raw := content

	// tolerates markdown fences or prose before the JSON
	if i := strings.Index(raw, "{"); i >= 0 {
		raw = raw[i:]
	}
	raw = escapeRawNewlinesInStrings(raw)

	var result Result

	// Decode (rather than Unmarshal) so e.g. a stray extra "}" small local models emits
	// don't turn an otherwise-valid answer into a parse error.
	dec := json.NewDecoder(strings.NewReader(raw))
	if err := dec.Decode(&result); err != nil {
		return Result{}, fmt.Errorf("agent: parse final answer %q: %w", content, err)
	}

	result.FinalCode = strings.TrimSpace(result.FinalCode)

	return result, nil
}

// escapeRawNewlinesInStrings replaces literal newline/tab bytes that fall
// inside a JSON string literal with their escaped form. Small local models
// often pretty-print their reason field with real line breaks instead of
// \n, which JSON's grammar forbids inside a string.
func escapeRawNewlinesInStrings(s string) string {
	var b strings.Builder
	inString, escaped := false, false
	for _, r := range s {
		if !inString {
			if r == '"' {
				inString = true
			}
			b.WriteRune(r)
			continue
		}
		if escaped {
			escaped = false
			b.WriteRune(r)
			continue
		}
		switch r {
		case '\\':
			escaped = true
			b.WriteRune(r)
		case '"':
			inString = false
			b.WriteRune(r)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\'':
			b.WriteString(`'`)
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}
