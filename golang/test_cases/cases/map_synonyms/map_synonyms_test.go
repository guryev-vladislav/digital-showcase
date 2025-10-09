package mapsynonyms

import "testing"

func TestMapSynonyms(t *testing.T) {
	cases := []struct {
		lines  []string
		target string
		want   string
		wantOk bool
	}{
		{
			lines:  []string{"hello world", "world foo"},
			target: "hello",
			want:   "world",
			wantOk: true,
		},
		{
			lines:  []string{"hello world", "world foo"},
			target: "foo",
			want:   "world",
			wantOk: true,
		},
		{
			lines:  []string{"hello world", "world foo"},
			target: "bar",
			want:   "",
			wantOk: false,
		},
	}

	for _, c := range cases {
		got, ok := MapSynonyms(c.lines, c.target)
		if got != c.want {
			t.Errorf("MapSynonyms(%v, %q) = %q, want %q", c.lines, c.target, got, c.want)
		}
		if ok != c.wantOk {
			t.Errorf("MapSynonyms(%v, %q) = (_, %v), want (_, %v)", c.lines, c.target, ok, c.wantOk)
		}
	}
}
