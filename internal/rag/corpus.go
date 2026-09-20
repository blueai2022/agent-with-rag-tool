package rag

import (
	"encoding/xml"
	"io"
	"strings"
)

// CodeMeta is one ICD-10-CM index entry: a billable code with the fields the
// grounding step needs. Description is the text that is embedded for retrieval;
// Display is an optional human-facing override shown to the selector instead of
// Description (used by acronym alias rows, whose embedded text is the terse
// acronym but whose display text should be the expanded condition).
type CodeMeta struct {
	Code           string
	Description    string // embedded text
	Display        string // display text; empty means use Description
	Category       string // 3-char category (e.g. "E11")
	Billable       bool
	InclusionTerms []string
	CodeAlso       []string
	Excludes1      []string
	Excludes2      []string
}

// DisplayDescription returns the text to show for an entry: Display when set,
// else Description.
func DisplayDescription(m CodeMeta) string {
	if m.Display != "" {
		return m.Display
	}

	return m.Description
}

// xmlDiag mirrors the tabular XML <diag> element. Nested <diag> children are
// sub-codes; inclusion/excludes/codeAlso notes are captured via their <note>
// children using the path tag syntax.
type xmlDiag struct {
	Name  string    `xml:"name"`
	Desc  string    `xml:"desc"`
	Incl  []string  `xml:"inclusionTerm>note"`
	Ex1   []string  `xml:"excludes1>note"`
	Ex2   []string  `xml:"excludes2>note"`
	Also  []string  `xml:"codeAlso>note"`
	Child []xmlDiag `xml:"diag"`
}

// ParseCorpus reads the ICD-10-CM tabular XML and returns the billable (leaf)
// codes, each resolved to a CodeMeta with its 3-char category.
func ParseCorpus(r io.Reader) ([]CodeMeta, error) {
	dec := xml.NewDecoder(r)

	var out []CodeMeta

	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}

		if err != nil {
			return nil, err
		}

		start, ok := tok.(xml.StartElement)
		if !ok || start.Name.Local != "diag" {
			continue
		}

		var d xmlDiag

		if err := dec.DecodeElement(&d, &start); err != nil {
			return nil, err
		}
		flattenDiag(&d, &out)
	}

	return out, nil
}

// flattenDiag walks a <diag> subtree and emits a CodeMeta for every leaf (a
// billable code with no children).
func flattenDiag(d *xmlDiag, out *[]CodeMeta) {
	if len(d.Child) == 0 {
		code := strings.TrimSpace(d.Name)
		if code == "" {
			return
		}
		*out = append(*out, CodeMeta{
			Code:           code,
			Description:    strings.TrimSpace(d.Desc),
			Category:       categoryOf(code),
			Billable:       true,
			InclusionTerms: trimNotes(d.Incl),
			CodeAlso:       trimNotes(d.Also),
			Excludes1:      trimNotes(d.Ex1),
			Excludes2:      trimNotes(d.Ex2),
		})
		return
	}
	for i := range d.Child {
		flattenDiag(&d.Child[i], out)
	}
}

func categoryOf(code string) string {
	if len(code) >= 3 {
		return code[:3]
	}
	return code
}

func trimNotes(notes []string) []string {
	var out []string
	for _, n := range notes {
		if s := strings.TrimSpace(n); s != "" {
			out = append(out, s)
		}
	}
	return out
}
