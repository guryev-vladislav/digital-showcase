package polyglots

import (
	"reflect"
	"testing"
)

func TestPolyglots(t *testing.T) {
	tests := []struct {
		name      string
		students  [][]string
		common    []string
		allUnique []string
	}{
		{
			name: "valid input",
			students: [][]string{
				{"English", "Russian", "French"},
				{"English", "Russian", "Spanish"},
				{"English", "Russian", "German"},
			},
			common:    []string{"English", "Russian"},
			allUnique: []string{"English", "French", "German", "Russian", "Spanish"},
		},
		{
			name: "nil common languages input",
			students: [][]string{
				{"English", "French"},
				{"Russian", "Spanish"},
				{"Korean", "German"},
			},
			common:    []string{},
			allUnique: []string{"English", "French", "German", "Korean", "Russian", "Spanish"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			polyglots := Polyglots(tt.students)
			if !reflect.DeepEqual(polyglots.commonLanguages, tt.common) {
				t.Errorf("Polyglots().commonLanguages = %v, want %v", polyglots.commonLanguages, tt.common)
			}
			if !reflect.DeepEqual(polyglots.allUniqueLanguages, tt.allUnique) {
				t.Errorf("Polyglots().allUniqueLanguages = %v, want %v", polyglots.allUniqueLanguages, tt.allUnique)
			}
		})
	}
}
