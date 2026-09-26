package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	llm "icd10-agent/internal/llm"
	"icd10-agent/internal/rag"
)

// flexInt unmarshals a JSON number OR a numeric string into an int — small
// local models frequently emit tool-call arguments as strings regardless of
// the declared JSON schema type.
type flexInt int

func (fi *flexInt) UnmarshalJSON(data []byte) error {
	var n int

	if err := json.Unmarshal(data, &n); err == nil {
		*fi = flexInt(n)

		return nil
	}

	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}

	n, err := strconv.Atoi(s)
	if err != nil {
		return fmt.Errorf("flexInt: %q is not a number: %w", s, err)
	}
	*fi = flexInt(n)

	return nil
}

// searchICD10Tool builds the search_icd10 tool schema and a dispatch func
// that runs it against c and always returns a string (errors are reported
// back to the model as tool output, not returned to the caller, so a bad
// argument doesn't abort the loop).
func searchICD10Tool(r *rag.Retriever) (llm.Tool, func(llm.ToolCall) string) {
	schema := llm.Tool{
		Type: "function",
		Function: llm.Function{
			Name:        "search_icd10",
			Description: "Search the ICD-10-CM sample corpus for candidate codes matching a short clinical phrase.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"query": {"type": "string", "description": "short clinical phrase to search for"},
					"k": {"type": "integer", "description": "number of candidates to return (default 5)"}
				},
				"required": ["query"]
			}`),
		},
	}

	dispatch := func(call llm.ToolCall) string {
		if call.Function.Name != "search_icd10" {
			return fmt.Sprintf(`{"error": "unknown tool %q"}`, call.Function.Name)
		}

		var args struct {
			Query string  `json:"query"`
			K     flexInt `json:"k"`
		}

		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			return fmt.Sprintf(`{"error": "invalid arguments: %s"}`, err)
		}

		args.Query = strings.TrimSpace(args.Query)
		if args.Query == "" {
			return `{"error": "missing required argument \"query\": a short clinical phrase to search for"}`
		}

		k := int(args.K)
		if k <= 0 {
			k = 5
		}

		results, err := r.Retrieve(context.TODO(), args.Query, k)
		if err != nil {
			return fmt.Sprintf(`{"error": "retrieve failed: %s"}`, err)
		}

		out, err := json.Marshal(results)
		if err != nil {
			return fmt.Sprintf(`{"error": "marshal results: %s"}`, err)
		}

		return string(out)
	}

	return schema, dispatch
}
