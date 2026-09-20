// Package corpus is a deliberately tiny "retrieval" tool: lexical
// token-overlap search over a small, hand-curated ICD-10-CM sample, standing
// in for the production system's embedding-based RAG index. The point of
// this repo is the agent shape (an LLM driving its own tool calls), not
// retrieval quality — see the rag-rerank-eval repo for that topic.
package corpus

import (
	"encoding/json"
	"os"
	"regexp"
	"sort"
	"strings"
)

// Code is one ICD-10-CM sample entry.
type Code struct {
	Code        string `json:"code"`
	Description string `json:"description"`
}

// Corpus is an in-memory, lexically searchable set of codes.
type Corpus struct {
	codes  []Code
	tokens [][]string // tokens[i] = tokenize(codes[i].Description)
}

var tokenRE = regexp.MustCompile(`[a-zA-Z0-9]+`)

func tokenize(s string) []string {
	return tokenRE.FindAllString(strings.ToLower(s), -1)
}

// Load reads a corpus JSON file (an array of {"code","description"} objects).
func Load(path string) (*Corpus, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var codes []Code
	if err := json.Unmarshal(data, &codes); err != nil {
		return nil, err
	}

	c := &Corpus{codes: codes, tokens: make([][]string, len(codes))}
	for i, code := range codes {
		c.tokens[i] = tokenize(code.Description)
	}

	return c, nil
}

// Search ranks codes by the number of query tokens found in their
// description (case-insensitive whole-word overlap), descending. Ties keep
// the corpus's original order.
func (c *Corpus) Search(query string, k int) []Code {
	if k <= 0 || len(c.codes) == 0 {
		return nil
	}

	qtokens := tokenize(query)
	qset := make(map[string]struct{}, len(qtokens))
	for _, t := range qtokens {
		qset[t] = struct{}{}
	}

	type scored struct {
		idx   int
		score int
	}

	scores := make([]scored, len(c.codes))
	for i, toks := range c.tokens {
		var s int
		for _, t := range toks {
			if _, ok := qset[t]; ok {
				s++
			}
		}
		scores[i] = scored{idx: i, score: s}
	}
	sort.SliceStable(scores, func(a, b int) bool { return scores[a].score > scores[b].score })

	if k > len(scores) {
		k = len(scores)
	}

	out := make([]Code, k)
	for i := 0; i < k; i++ {
		out[i] = c.codes[scores[i].idx]
	}

	return out
}

// Lookup returns a code's description by exact (case-insensitive) code match.
func (c *Corpus) Lookup(code string) (Code, bool) {
	for _, cd := range c.codes {
		if strings.EqualFold(cd.Code, code) {
			return cd, true
		}
	}
	return Code{}, false
}
