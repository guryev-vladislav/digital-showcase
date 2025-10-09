package signalstations

import (
	"sort"
)

func SignalStations(signal1, signal2, signal3 int) int {
	a := signal1 + signal2
	b := signal3 + signal2
	c := signal3 + signal1
	sides := []int{a, b, c}
	sort.Ints(sides)

	if sides[0]+sides[1] <= sides[2] {
		return -1
	}

	s := (a + b + c) / 2
	area := (s * (s - a) * (s - b) * (s - c))

	return area
}
