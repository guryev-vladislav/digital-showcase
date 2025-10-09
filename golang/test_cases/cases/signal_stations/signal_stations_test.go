package signalstations

import "testing"

func TestSignalStations(t *testing.T) {
	tests := []struct {
		name     string
		signal1  int
		signal2  int
		signal3  int
		expected int
	}{
		{
			name:     "valid triangle",
			signal1:  3,
			signal2:  4,
			signal3:  5,
			expected: 720,
		},
		{
			name:     "valid triangle",
			signal1:  1,
			signal2:  2,
			signal3:  5,
			expected: 80,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual := SignalStations(tt.signal1, tt.signal2, tt.signal3)
			if actual != tt.expected {
				t.Errorf("SignalStations() = %v, want %v", actual, tt.expected)
			}
		})
	}
}
