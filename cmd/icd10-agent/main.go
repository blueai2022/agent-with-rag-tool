// Command icd10-agent is a CLI demo of the agent: given a quoted piece of
// clinical evidence and an upstream candidate code, it lets an LLM drive its
// own search_icd10 tool calls before returning a final ICD-10 code.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"icd10-agent/internal/agent"
	"icd10-agent/internal/llm"
	"icd10-agent/internal/rag"
)

func main() {
	var (
		quoted          = flag.String("quoted", "", "quoted clinical evidence text (required)")
		code            = flag.String("code", "", "upstream/candidate ICD-10 code")
		baseURL         = flag.String("base-url", envOr("ICD10_AGENT_BASE_URL", "http://localhost:11434/v1"), "OpenAI-compatible chat-completions base URL")
		apiKey          = flag.String("api-key", os.Getenv("OPENAI_API_KEY"), "API key, if the endpoint requires one")
		model           = flag.String("model", envOr("ICD10_AGENT_MODEL", "llama3.1:8b"), "chat model id (must support tool calling)")
		ragIndex        = flag.String("rag-index", "icd_index.bin", "path to the built RAG index artifact")
		promptPath      = flag.String("prompt", "prompts/system.txt", "path to the system prompt")
		verbose         = flag.Bool("v", false, "print the full tool-call transcript to stderr")
		ragEmbedBaseURL = flag.String("rag-embed-base-url", "http://localhost:11434", "RAG embeddings base URL")
		ragEmbedModel   = flag.String("rag-embed-model", "mxbai-embed-large", "RAG embedding model")
	)
	flag.Parse()

	if *quoted == "" {
		log.Fatal("icd10-agent: --quoted is required")
	}

	index, err := rag.LoadIndex(*ragIndex)
	if err != nil {
		log.Fatalf("icd10-agent: load RAG index: %v", err)
	}

	embedder := rag.NewOllamaEmbedder(*ragEmbedBaseURL, *ragEmbedModel, "")
	retriever := rag.NewRetriever(embedder, index)

	systemPrompt, err := os.ReadFile(*promptPath)
	if err != nil {
		log.Fatalf("icd10-agent: read system prompt: %v", err)
	}

	llmClient := llm.New(*baseURL, *apiKey, *model, nil)
	a := agent.New(llmClient, retriever, string(systemPrompt))

	result, transcript, err := a.Select(context.Background(), *quoted, *code)
	if *verbose {
		for _, m := range transcript {
			fmt.Fprintf(os.Stderr, "--- %s ---\n", m.Role)
			if m.Content != "" {
				fmt.Fprintln(os.Stderr, m.Content)
			}
			for _, tc := range m.ToolCalls {
				fmt.Fprintf(os.Stderr, "tool_call: %s(%s)\n", tc.Function.Name, tc.Function.Arguments)
			}
		}
	}
	if err != nil {
		log.Fatalf("icd10-agent: %v", err)
	}

	out, _ := json.MarshalIndent(result, "", "  ")
	fmt.Println(string(out))
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}

	return fallback
}
