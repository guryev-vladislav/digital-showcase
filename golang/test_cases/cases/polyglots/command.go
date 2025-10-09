package polyglots

import (
	"bufio"
	"fmt"
	"os"
	"strconv"

	"github.com/spf13/cobra"
)

var PolyglotsCmd = &cobra.Command{
	Use:   "polyglots",
	Short: "Анализ языков, которые знают школьники",
	Long: `Команда polyglots анализирует, какие языки знают все школьники 
	и какие языки знает хотя бы один школьник.

	Формат ввода:
	- Первая строка: количество школьников N (1 ≤ N ≤ 1000)
	- Для каждого школьника:
	* количество языков M[i] (1 ≤ M[i] ≤ 500)
	* M[i] строк с названиями языков (длина названий до 1000 символов, всего языков до 1000)

	Формат вывода:
	- Количество языков, которые знают все школьники
	- Список этих языков (по одному на строку)
	- Количество языков, которые знает хотя бы один школьник
	- Список этих языков (по одному на строку)

	Пример использования:
	echo -e "3\n2\nEnglish\nRussian\n1\nEnglish\n2\nRussian\nFrench" | go run main.go polyglots`,
	Run: polyglotsCmdRun,
}

func polyglotsCmdRun(cmd *cobra.Command, args []string) {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	n, _ := strconv.Atoi(scanner.Text())

	studentLanguages := make([][]string, n)

	for i := range n {
		scanner.Scan()
		m, _ := strconv.Atoi(scanner.Text())
		studentLanguages[i] = make([]string, m)
		for j := range m {
			scanner.Scan()
			language := scanner.Text()
			studentLanguages[i][j] = language
		}
	}

	polyglots := Polyglots(studentLanguages)

	fmt.Println(len(polyglots.commonLanguages))
	for _, language := range polyglots.commonLanguages {
		fmt.Println(language)
	}

	fmt.Println(len(polyglots.allUniqueLanguages))
	for _, language := range polyglots.allUniqueLanguages {
		fmt.Println(language)
	}
}
