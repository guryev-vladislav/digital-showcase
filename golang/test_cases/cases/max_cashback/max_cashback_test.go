package maxcashback

import "testing"

func TestMaxCashBack(t *testing.T) {
	tests := []struct {
		name     string
		min      int
		cashback int
		sum      int
		expected int
	}{
		{
			name:     "valid input",
			min:      100,
			cashback: 5,
			sum:      450,
			expected: 20,
		},
		{
			name:     "invalid input",
			min:      500,
			cashback: 5,
			sum:      450,
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual := MaxCashBack(tt.min, tt.cashback, tt.sum)
			if actual != tt.expected {
				t.Errorf("MaxCashBack() = %v, want %v", actual, tt.expected)
			}
		})
	}
}
