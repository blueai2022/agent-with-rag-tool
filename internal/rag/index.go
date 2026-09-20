package rag

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// Candidate is one retrieval result.
type Candidate struct {
	Code        string
	Description string
	CodeAlso    []string
	Score       float32
}

// Index is an in-memory vector index over billable ICD codes.
type Index struct {
	dim   int
	codes []string
	meta  []CodeMeta
	vecs  []float32 // len(codes)*dim, row-major
	norms []float32 // ||vecs[i]||
}

// NewIndex builds an index from codes, their metadata, and a row-major vector
// slice (len(codes)*dim).
func NewIndex(dim int, codes []string, meta []CodeMeta, vecs []float32) *Index {
	n := len(codes)
	norms := make([]float32, n)
	for i := range n {
		norms[i] = norm(vecs[i*dim : (i+1)*dim])
	}

	return &Index{dim: dim, codes: codes, meta: meta, vecs: vecs, norms: norms}
}

func (ix *Index) Len() int { return len(ix.codes) }
func (ix *Index) Dim() int { return ix.dim }

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
	qnorm := norm(query)

	type scored struct {
		idx   int
		score float32
	}
	scores := make([]scored, len(ix.codes))
	for i := range ix.codes {
		s := float32(0)
		if qnorm != 0 && ix.norms[i] != 0 {
			s = dot(query, ix.vecs[i*ix.dim:(i+1)*ix.dim]) / (qnorm * ix.norms[i])
		}
		scores[i] = scored{idx: i, score: s}
	}
	sort.Slice(scores, func(a, b int) bool { return scores[a].score > scores[b].score })

	k = min(k, len(scores))

	out := make([]Candidate, k)
	for i := 0; i < k; i++ {
		m := ix.meta[scores[i].idx]
		out[i] = Candidate{
			Code:        m.Code,
			Description: DisplayDescription(m),
			CodeAlso:    m.CodeAlso,
			Score:       scores[i].score,
		}
	}
	return out, nil
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func norm(v []float32) float32 {
	var s float32
	for _, x := range v {
		s += x * x
	}
	return float32(math.Sqrt(float64(s)))
}
