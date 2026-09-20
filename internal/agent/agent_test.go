package agent

import "testing"

func TestParseResult(t *testing.T) {
	cases := []struct {
		name    string
		content string
		want    Result
	}{
		{
			name:    "plain json",
			content: `{"final_code": "E11.9", "reason": "ok"}`,
			want:    Result{FinalCode: "E11.9", Reason: "ok"},
		},
		{
			name:    "markdown fence prefix",
			content: "```json\n{\"final_code\": \"E11.9\", \"reason\": \"ok\"}\n```",
			want:    Result{FinalCode: "E11.9", Reason: "ok"},
		},
		{
			name:    "trailing extra brace",
			content: `{"final_code": "E11.40", "reason": "ok"}}`,
			want:    Result{FinalCode: "E11.40", Reason: "ok"},
		},
		{
			name: "literal newlines inside string value",
			content: "{\n  \"final_code\": \"E11.42\",\n  \"reason\": \"line one, \n" +
				"              line two.\"\n}",
			want: Result{FinalCode: "E11.42", Reason: "line one, \n              line two."},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := parseResult(c.content)
			if err != nil {
				t.Fatalf("parseResult(%q) returned error: %v", c.content, err)
			}
			if got != c.want {
				t.Errorf("parseResult(%q) = %+v, want %+v", c.content, got, c.want)
			}
		})
	}
}
