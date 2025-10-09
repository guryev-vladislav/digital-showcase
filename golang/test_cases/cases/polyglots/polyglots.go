package polyglots

import (
	"sort"
)

type PolyglotsLanguages struct {
	commonLanguages    []string
	allUniqueLanguages []string
}

func Polyglots(studentLanguages [][]string) *PolyglotsLanguages {

	allLanguages := make(map[string]int)

	for i := range len(studentLanguages) {
		for _, language := range studentLanguages[i] {

			allLanguages[language]++
		}
	}

	commonLanguages := []string{}
	for language, count := range allLanguages {
		if count == len(studentLanguages) {
			commonLanguages = append(commonLanguages, language)
		}
	}

	allUniqueLanguages := []string{}
	for language := range allLanguages {
		allUniqueLanguages = append(allUniqueLanguages, language)
	}

	sort.Strings(commonLanguages)
	sort.Strings(allUniqueLanguages)

	polyglots := PolyglotsLanguages{
		commonLanguages:    commonLanguages,
		allUniqueLanguages: allUniqueLanguages,
	}

	return &polyglots
}
