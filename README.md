# icd10-agent

An agent use RAG call as a tool to ensure upstream agent does not have
a "memory" mistake - an invalid ICD10 code or an imprecise one.

## The contrast this repo is making

A production RAG + LLM-selector pipeline (the shape this repo is exported
from) typically looks like this:

```mermaid
flowchart LR
    Q[quoted text] --> E[embed + cosine search]
    E --> L[fixed top-k lineup]
    L --> P[one prompt: quote + upstream code + full lineup]
    P --> LLM[single LLM call]
    LLM --> R["{final_code, reason}"]
```

Neither shape is strictly "better" — the fixed pipeline is cheaper, faster,
and fully deterministic (good for something like insurance underwriting);
the agent loop is more flexible and can recover from a bad first query, at
the cost of more LLM round-trips and less predictable behavior. This repo
exists to make that shape difference concrete and runnable.

## What's in here

- `internal/rag` — the retrieval package behind `search_icd10`: an
  `Embedder`/`Index`/`Retriever` do embedding-based cosine-similarity search
  (no external vector DB), plus acronym-alias enrichment and a binary
  artifact format so `cmd/rag-build` embeds the corpus once offline and the
  agent just loads the prebuilt index at runtime instead of re-embedding it
  on every run.
- `internal/llmclient` — a minimal OpenAI-compatible chat-completions client
  with tool calling (~100 lines, no SDK dependency).
- `internal/agent` — the loop itself: send messages + tool schema, execute
  any tool calls the model makes, append results, repeat until the model
  answers with JSON instead of a tool call (bounded by a max-steps guard).
- `cmd/icd10-agent` — CLI: `--quoted`, `--code`, and flags to point at any
  OpenAI-compatible endpoint.
- `cmd/rag-build` — offline build step: embeds `data/icd10_sample.json` (+
  alias rows) and writes the `icd_index.bin` artifact `cmd/icd10-agent`
  loads at runtime (see `make rag-index`).
- `prompts/system.txt` — the system prompt, adapted from the production
  selector's prompt to describe tool use instead of a pre-built lineup.

## Running it

Needs an OpenAI-compatible `/chat/completions` endpoint with **tool-calling
support**. Two easy options:

**Local, via Ollama** (no API key, but the model must support tools):

```bash
ollama pull llama3.1:8b   # or any tool-calling capable model
go run ./cmd/icd10-agent -quoted "diabetic neuropathy" -code "E11.9" \
  -model llama3.1:8b -v
```

**OpenAI**:

```bash
go run ./cmd/icd10-agent -quoted "DM2 with neuropathy" -code "E11.9" \
  -base-url https://api.openai.com/v1 -model gpt-4o-mini \
  -api-key "$OPENAI_API_KEY"
```

`-v` prints the full transcript (system/user/assistant/tool messages,
including every tool call) to stderr, so you can see the loop happen.

### Example transcript (llama3.1:8b, local)

```
--- user ---
Quoted evidence: diabetic neuropathy
Upstream code: E11.9
--- assistant ---
tool_call: search_icd10({"query":"diabetic nerve damage","k":"10"})
--- tool ---
[{"code":"E11.42","description":"Type 2 diabetes mellitus with diabetic polyneuropathy"}, ...]
--- assistant ---
{"final_code": "E11.9", "reason": "..."}
```

The model chose its own search phrasing ("diabetic nerve damage", not the
literal input text), got back candidates, and only then answered. Answer
*quality* depends heavily on the model. An 8B-class model (`llama3.1:8b`
above) is the recommended minimum for reliable results: it converges in a
single search + answer, with valid JSON and a code that's actually
justified by the evidence.

Smaller models (e.g. `llama3.2:1b`) still demonstrate the tool-calling
*mechanics* — they'll call `search_icd10` and eventually stop — but in
practice they're unreliable at the *reasoning*.

## Makefile targets

- `make build` — `go build ./...`
- `make test` — `go test ./...`
- `make rag-index` — embeds `data/icd10_sample.json` (+
  `internal/rag/data/aliases.json`) and writes the `icd_index.bin` artifact
  the agent loads at runtime.
- `make run` — runs the agent CLI; override defaults on the command line,
  e.g. `make run QUOTED="chest pain" CODE=R07.9`.
- `make clean` — removes `icd_index.bin`.

## Data

`data/icd10_sample.json` — ~50 ICD-10-CM codes and their official short
titles (a US government work, not subject to copyright), covering common
conditions plus near-neighbor distractors. This is a small hand-curated
educational subset, **not** a complete or authoritative code set — do not
use it for actual clinical coding or billing.
