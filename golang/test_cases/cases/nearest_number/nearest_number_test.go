package nearestnumber

import "testing"

func TestNearestNumber(t *testing.T) {
	tests := []struct {
		elements []int
		x        int
		expected int
	}{
		{[]int{1, 2, 3, 4, 5}, 3, 3},
		{[]int{1, 2, 3, 4, 5}, 4, 4},
		{[]int{1, 2, 3, 4, 5}, 6, 5},
		{[]int{1, 2, 3, 4, 5}, 7, 5},
		{[]int{1, 2, 3, 4, 5}, 10, 5},
	}

	for _, test := range tests {
		actual := NearestNumber(test.elements, test.x)
		if actual != test.expected {
			t.Errorf("NearestNumber(%v, %d) = %d, but got %d", test.elements, test.x, test.expected, actual)
		}
	}
}
