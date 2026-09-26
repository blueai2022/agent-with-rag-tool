package rag

import (
	"fmt"
	"math"
	"strings"

	faiss "github.com/DataIntelligenceCrew/go-faiss"
)

// Candidate is one retrieval result.
type Candidate struct {
	Code        string
	Description string
	CodeAlso    []string
	Score       float32
}

// Index is an in-memory vector index over billable ICD codes, backed by a
// Faiss flat inner-product index over L2-normalized vectors (equivalent to
// exact cosine-similarity search).
type Index struct {
	dim   int
	codes []string
	meta  []CodeMeta
	vecs  []float32 // len(codes)*dim, row-major
	faiss faiss.Index
}

// NewIndex builds an index from codes, their metadata, and a row-major vector
// slice (len(codes)*dim).
func NewIndex(dim int, codes []string, meta []CodeMeta, vecs []float32) (*Index, error) {
	n := len(codes)

	idx, err := faiss.NewIndexFlatIP(dim)
	if err != nil {
		return nil, fmt.Errorf("rag: create faiss index: %w", err)
	}

	if n > 0 {
		normalized := make([]float32, len(vecs))
		copy(normalized, vecs)
		for i := range n {
			l2Normalize(normalized[i*dim : (i+1)*dim])
		}

		if err := idx.Add(normalized); err != nil {
			return nil, fmt.Errorf("rag: add vectors to faiss index: %w", err)
		}
	}

	return &Index{
		dim:   dim,
		codes: codes,
		meta:  meta,
		vecs:  vecs,
		faiss: idx,
	}, nil
}

// Close releases the memory held by the underlying Faiss index.
func (ix *Index) Close() {
	ix.faiss.Delete()
}

func (ix *Index) Len() int {
	return len(ix.codes)
}

func (ix *Index) Dim() int {
	return ix.dim
}

// Lookup returns the metadata for a code, case-insensitively, if present in the
// index. It is used by the pipeline to render ①'s original code as an
// unlabeled candidate in the selection lineup.
func (ix *Index) Lookup(code string) (CodeMeta, bool) {
	for i, c := range ix.codes {
		if strings.EqualFold(c, code) {
			return ix.meta[i], true
		}
	}

	return CodeMeta{}, false
}

// Search returns the top-k codes by cosine similarity to the query vector.
func (ix *Index) Search(query []float32, k int) ([]Candidate, error) {
	if len(query) != ix.dim {
		return nil, fmt.Errorf("rag: query dim %d != index dim %d", len(query), ix.dim)
	}

	if k <= 0 || len(ix.codes) == 0 {
		return nil, nil
	}

	k = min(k, len(ix.codes))

	normalizedQuery := make([]float32, len(query))
	copy(normalizedQuery, query)
	l2Normalize(normalizedQuery)

	scores, labels, err := ix.faiss.Search(normalizedQuery, int64(k))
	if err != nil {
		return nil, fmt.Errorf("rag: faiss search: %w", err)
	}

	out := make([]Candidate, 0, k)
	for i, label := range labels {
		if label < 0 {
			continue
		}
		m := ix.meta[label]
		out = append(out, Candidate{
			Code:        m.Code,
			Description: DisplayDescription(m),
			CodeAlso:    m.CodeAlso,
			Score:       scores[i],
		})
	}

	return out, nil
}

// l2Normalize scales v to unit L2 norm, leaving zero vectors unchanged.
func l2Normalize(v []float32) {
	var s float32
	for _, x := range v {
		s += x * x
	}

	if s == 0 {
		return
	}

	n := float32(math.Sqrt(float64(s)))
	for i := range v {
		v[i] /= n
	}
}
