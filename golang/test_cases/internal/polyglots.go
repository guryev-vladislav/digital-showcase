package internal

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strconv"

	"github.com/spf13/cobra"
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

func Polyglots(cmd *cobra.Command, args []string) {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	n, _ := strconv.Atoi(scanner.Text())

	allLanguages := make(map[string]int)
	studentLanguages := make([][]string, n)

	for i := range n {
		scanner.Scan()
		m, _ := strconv.Atoi(scanner.Text())
		studentLanguages[i] = make([]string, m)
		for j := range m {
			scanner.Scan()
			language := scanner.Text()
			studentLanguages[i][j] = language
			allLanguages[language]++
		}
	}

	commonLanguages := []string{}
	for language, count := range allLanguages {
		if count == n {
			commonLanguages = append(commonLanguages, language)
		}
	}

	allUniqueLanguages := []string{}
	for language := range allLanguages {
		allUniqueLanguages = append(allUniqueLanguages, language)
	}

	sort.Strings(commonLanguages)
	sort.Strings(allUniqueLanguages)

	fmt.Println(len(commonLanguages))
	for _, language := range commonLanguages {
		fmt.Println(language)
	}

	fmt.Println(len(allUniqueLanguages))
	for _, language := range allUniqueLanguages {
		fmt.Println(language)
	}
}
