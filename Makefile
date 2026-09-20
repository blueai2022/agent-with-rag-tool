.PHONY: build vet test rag-index run clean

# Ollama (or any OpenAI-compatible endpoint) defaults; override on the command
# line, e.g. `make run QUOTED="chest pain" CODE=R07.9`.
BASE_URL        ?= http://localhost:11434/v1
MODEL           ?= llama3.1:8b
EMBED_BASE_URL  ?= http://localhost:11434
EMBED_MODEL     ?= mxbai-embed-large
RAG_INDEX       ?= icd_index.bin
QUOTED          ?= DM2 with neuropathy
CODE             ?= E11.9

build:
	go build ./...

test:
	go test ./...

# Embeds data/icd10_sample.json (+ internal/rag/data/aliases.json) and writes
# the index artifact the agent loads at runtime.
rag-index:
	rm -f $(RAG_INDEX)
	go run ./cmd/rag-build -base-url $(EMBED_BASE_URL) -model $(EMBED_MODEL) -out $(RAG_INDEX)

run:
	go run ./cmd/icd10-agent -quoted "$(QUOTED)" -code "$(CODE)" \
		-base-url $(BASE_URL) -model $(MODEL) -rag-index $(RAG_INDEX) -v

clean:
	rm -f $(RAG_INDEX)
