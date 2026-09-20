package rag

import (
	"encoding/json"
	"os"
)

// Alias is one acronym/short-hand entry that is embedded into the index as an
// additional row pointing at a real billable code. Acronym is the embedded text
// (e.g. "HTN"); Code is the target code (e.g. "I10"); Name is the expanded
// condition used for the display description ("Hypertension - HTN").
type Alias struct {
	Acronym string `json:"acronym"`
	Code    string `json:"code"`
	Name    string `json:"name"`
}

// LoadAliases reads the alias data file (a JSON array of Alias).
func LoadAliases(path string) ([]Alias, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var aliases []Alias

	if err := json.Unmarshal(data, &aliases); err != nil {
		return nil, err
	}

	return aliases, nil
}

// AliasCodeMetas converts aliases into index rows: Description is the acronym
// (embedded text), Display is "<Name> - <Acronym>" (selector-facing), and the
// row points at the alias's target code. The target code is not validated here;
// the build step verifies it resolves to a real index entry.
func AliasCodeMetas(aliases []Alias) []CodeMeta {
	out := make([]CodeMeta, 0, len(aliases))
	for _, a := range aliases {
		out = append(out, CodeMeta{
			Code:        a.Code,
			Description: a.Acronym,
			Display:     a.Name + " - " + a.Acronym,
			Category:    categoryOf(a.Code),
			Billable:    true,
		})
	}
	return out
}
