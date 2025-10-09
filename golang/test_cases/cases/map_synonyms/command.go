package mapsynonyms

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
)

var MapSynonymsCmd = &cobra.Command{
	Use:   "map_synonyms",
	Short: "Найти синоним для заданного слова в словаре",
	Long: `Команда map_synonyms позволяет найти синоним для заданного слова в словаре синонимов.

	Вам дан словарь, состоящий из пар слов. Каждое слово является синонимом к парному ему слову.
	Все слова в словаре различны. Для одного данного слова определите его синоним.

	Формат ввода:
	- Программа получает на вход количество пар синонимов N
	- Далее следует N строк, каждая строка содержит ровно два слова-синонима
	- После этого следует одно слово

	Формат вывода:
	- Программа выводит синоним к данному слову

	Пример использования:
	echo -e "3\nhello hi\nworld earth\nbig large\nhello" | go run main.go map_synonyms`,
	Run: mapSynonymsCmdRun,
}

func mapSynonymsCmdRun(cmd *cobra.Command, args []string) {
	reader := bufio.NewReader(os.Stdin)

	var n int
	fmt.Fscan(reader, &n)
	reader.ReadString('\n')

	lines := make([]string, n)
	for i := range n {
		lines[i], _ = reader.ReadString('\n')
	}

	targetWord, _ := reader.ReadString('\n')
	targetWord = strings.TrimSpace(targetWord)

	synonym, ok := MapSynonyms(lines, targetWord)
	if ok {
		fmt.Println(synonym)
	} else {
		fmt.Println("None")
	}
}
