package rag

import "context"

// Retriever embeds a query and searches the index.
type Retriever struct {
	Embed Embedder
	Index *Index
}

func NewRetriever(e Embedder, ix *Index) *Retriever {
	return &Retriever{Embed: e, Index: ix}
}

// Retrieve embeds the query and returns the top-k candidates.
func (r *Retriever) Retrieve(ctx context.Context, query string, k int) ([]Candidate, error) {
	vecs, err := r.Embed.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	return r.Index.Search(vecs[0], k)
}
