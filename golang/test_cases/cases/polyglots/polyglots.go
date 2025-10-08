package polyglots

import (
	"sort"
)

/*
Каждый из N школьников некоторой школы знает M[i] языков.
Определите, какие языки знают все школьники и языки, которые знает хотя бы один из школьников.

Формат ввода
Первая строка входных данных содержит количество школьников N.
Далее идет N чисел M[i], после каждого из чисел идет M[i] строк, содержащих названия языков,
которые знает i-й школьник. Длина названий языков не превышает 1000 символов,
количество различных языков не более 1000. 1≤N≤1000, 1≤M[i]≤500.

Формат вывода
В первой строке выведите количество языков, которые знают все школьники.
Начиная со второй строки - список таких языков.
Затем - количество языков, которые знает хотя бы один школьник, на следующих строках - список таких языков.
*/

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
