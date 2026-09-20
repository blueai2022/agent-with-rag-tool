package rag

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

var artifactMagic = [4]byte{'R', 'A', 'G', '1'}

const artifactVersion = 2

// WriteArtifact writes the index (and its model) as a binary artifact:
// a header (magic, version, dim, count, model) followed by per-code records
// (length-prefixed string metadata + a raw float32 vector).
func WriteArtifact(w io.Writer, model string, ix *Index) error {
	if err := binary.Write(w, binary.LittleEndian, artifactMagic); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(artifactVersion)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(ix.dim)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint64(len(ix.codes))); err != nil {
		return err
	}
	if err := writeString(w, model); err != nil {
		return err
	}
	for i := range ix.codes {
		m := ix.meta[i]
		for _, s := range []string{m.Code, m.Description, m.Display, m.Category} {
			if err := writeString(w, s); err != nil {
				return err
			}
		}
		if err := binary.Write(w, binary.LittleEndian, m.Billable); err != nil {
			return err
		}
		for _, ss := range [][]string{m.InclusionTerms, m.CodeAlso, m.Excludes1, m.Excludes2} {
			if err := writeStrings(w, ss); err != nil {
				return err
			}
		}
		for _, x := range ix.vecs[i*ix.dim : (i+1)*ix.dim] {
			if err := binary.Write(w, binary.LittleEndian, x); err != nil {
				return err
			}
		}
	}
	return nil
}

// ReadArtifact reads an artifact and returns its model and index.
func ReadArtifact(r io.Reader) (model string, ix *Index, err error) {
	var magic [4]byte
	if err = binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return "", nil, err
	}
	if magic != artifactMagic {
		return "", nil, fmt.Errorf("rag: bad artifact magic")
	}
	var version, dim uint32
	if err = binary.Read(r, binary.LittleEndian, &version); err != nil {
		return "", nil, err
	}
	if version != artifactVersion {
		return "", nil, fmt.Errorf("rag: unsupported artifact version %d (want %d)", version, artifactVersion)
	}
	if err = binary.Read(r, binary.LittleEndian, &dim); err != nil {
		return "", nil, err
	}
	var count uint64
	if err = binary.Read(r, binary.LittleEndian, &count); err != nil {
		return "", nil, err
	}
	if model, err = readString(r); err != nil {
		return "", nil, err
	}

	codes := make([]string, count)
	meta := make([]CodeMeta, count)
	vecs := make([]float32, count*uint64(dim))
	for i := uint64(0); i < count; i++ {
		m := CodeMeta{}
		if m.Code, err = readString(r); err != nil {
			return "", nil, err
		}
		if m.Description, err = readString(r); err != nil {
			return "", nil, err
		}
		if m.Display, err = readString(r); err != nil {
			return "", nil, err
		}
		if m.Category, err = readString(r); err != nil {
			return "", nil, err
		}
		if err = binary.Read(r, binary.LittleEndian, &m.Billable); err != nil {
			return "", nil, err
		}
		if m.InclusionTerms, err = readStrings(r); err != nil {
			return "", nil, err
		}
		if m.CodeAlso, err = readStrings(r); err != nil {
			return "", nil, err
		}
		if m.Excludes1, err = readStrings(r); err != nil {
			return "", nil, err
		}
		if m.Excludes2, err = readStrings(r); err != nil {
			return "", nil, err
		}
		codes[i] = m.Code
		meta[i] = m
		base := i * uint64(dim)
		for j := uint32(0); j < dim; j++ {
			var x float32
			if err = binary.Read(r, binary.LittleEndian, &x); err != nil {
				return "", nil, err
			}
			vecs[base+uint64(j)] = x
		}
	}
	return model, NewIndex(int(dim), codes, meta, vecs), nil
}

// LoadIndex loads an index from a file path.
func LoadIndex(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	_, ix, err := ReadArtifact(f)
	return ix, err
}

func writeString(w io.Writer, s string) error {
	b := []byte(s)
	if err := binary.Write(w, binary.LittleEndian, uint32(len(b))); err != nil {
		return err
	}
	_, err := w.Write(b)
	return err
}

func writeStrings(w io.Writer, ss []string) error {
	if err := binary.Write(w, binary.LittleEndian, uint32(len(ss))); err != nil {
		return err
	}
	for _, s := range ss {
		if err := writeString(w, s); err != nil {
			return err
		}
	}
	return nil
}

func readString(r io.Reader) (string, error) {
	var n uint32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return "", err
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(r, b); err != nil {
		return "", err
	}
	return string(b), nil
}

func readStrings(r io.Reader) ([]string, error) {
	var n uint32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	out := make([]string, n)
	for i := range out {
		s, err := readString(r)
		if err != nil {
			return nil, err
		}
		out[i] = s
	}
	return out, nil
}
