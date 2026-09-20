package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"icd10-agent/internal/rag"
)

func main() {
	var (
		jsonPath  = flag.String("json", "data/icd10_sample.json", "path to the sample ICD-10-CM JSON corpus")
		outPath   = flag.String("out", "icd_index.bin", "output artifact path")
		aliasPath = flag.String("aliases", "internal/rag/data/aliases.json", "path to acronym alias JSON (empty to skip)")
		baseURL   = flag.String("base-url", "http://localhost:11434", "embeddings base URL")
		model     = flag.String("model", "mxbai-embed-large", "embedding model")
		chunk     = flag.Int("chunk", 64, "embedding batch size")
	)
	flag.Parse()

	meta, err := loadSampleCorpus(*jsonPath)
	if err != nil {
		log.Fatalf("load corpus: %v", err)
	}
	if len(meta) == 0 {
		log.Fatal("corpus parsed to zero billable codes")
	}

	// Acronym aliases are appended as extra index rows (embedded text = acronym,
	// display = "<Name> - <Acronym>"), so an acronym query matches its row
	// exactly. Validate each target code resolves to a real billable code.
	if *aliasPath != "" {
		aliases, aerr := rag.LoadAliases(*aliasPath)
		if aerr != nil {
			log.Fatalf("load aliases: %v", aerr)
		}
		known := make(map[string]bool, len(meta))
		for _, m := range meta {
			known[m.Code] = true
		}
		for _, a := range aliases {
			if !known[a.Code] {
				log.Fatalf("alias %q targets unknown code %q", a.Acronym, a.Code)
			}
		}
		meta = append(meta, rag.AliasCodeMetas(aliases)...)
		log.Printf("appended %d alias rows", len(aliases))
	}

	codes := make([]string, len(meta))
	descs := make([]string, len(meta))
	for i := range meta {
		codes[i] = meta[i].Code
		descs[i] = meta[i].Description
	}

	emb := rag.NewOllamaEmbedder(*baseURL, *model, "")
	ctx := context.Background()

	// Embed in chunks; the first chunk reveals the model's dimension.
	var vecs []float32
	dim := 0
	for start := 0; start < len(meta); start += *chunk {
		end := start + *chunk

		end = min(end, len(meta))

		out, err := emb.Embed(ctx, descs[start:end])
		if err != nil {
			log.Fatalf("embed %d..%d: %v", start, end, err)
		}

		if dim == 0 {
			if len(out) == 0 || len(out[0]) == 0 {
				log.Fatal("embedder returned empty vectors")
			}
			dim = len(out[0])
			vecs = make([]float32, len(meta)*dim)
		}

		for i, v := range out {
			if len(v) != dim {
				log.Fatalf("dimension changed mid-build: %d != %d", len(v), dim)
			}
			copy(vecs[(start+i)*dim:], v)
		}
		log.Printf("embedded %d/%d codes", end, len(meta))
	}

	ix := rag.NewIndex(dim, codes, meta, vecs)

	out, err := os.Create(*outPath)
	if err != nil {
		log.Fatalf("create artifact: %v", err)
	}
	defer out.Close()

	if err := rag.WriteArtifact(out, *model, ix); err != nil {
		log.Fatalf("write artifact: %v", err)
	}
	fmt.Printf("wrote %d codes (dim %d, model %s) to %s\n", len(meta), dim, *model, *outPath)
}

// sampleEntry mirrors one row of the hand-curated data/icd10_sample.json
// corpus: just a code and its short title, no tabular-XML notes/hierarchy.
type sampleEntry struct {
	Code        string `json:"code"`
	Description string `json:"description"`
}

// loadSampleCorpus reads the sample JSON corpus and adapts each entry into a
// rag.CodeMeta (all entries are treated as billable leaf codes).
func loadSampleCorpus(path string) ([]rag.CodeMeta, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var entries []sampleEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, err
	}
	meta := make([]rag.CodeMeta, len(entries))
	for i, e := range entries {
		category := e.Code
		if len(category) > 3 {
			category = category[:3]
		}
		meta[i] = rag.CodeMeta{
			Code:        e.Code,
			Description: e.Description,
			Category:    category,
			Billable:    true,
		}
	}
	return meta, nil
}
