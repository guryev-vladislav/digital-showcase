package solutions

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

/*
Вам дан словарь, состоящий из пар слов. Каждое слово является синонимом к парному ему слову.
Все слова в словаре различны. Для одного данного слова определите его синоним.

Формат ввода
Программа получает на вход количество пар синонимов N.Далее следует N строк,
каждая строка содержит ровно два слова-синонима. После этого следует одно слово.

Формат вывода
Программа должна вывести синоним к данному слову.
*/

func MapSynonyms() {
	reader := bufio.NewReader(os.Stdin)

	var n int
	fmt.Println("Enter the number of synonyms:")
	fmt.Fscan(reader, &n)
	reader.ReadString('\n')

	synonyms := make(map[string]string, n*2)
	fmt.Println("Enter the synonyms:")
	for range n {
		line, _ := reader.ReadString('\n')
		words := strings.Split(strings.TrimSpace(line), " ")
		word1, word2 := words[0], words[1]
		synonyms[word1] = word2
		synonyms[word2] = word1
	}

	fmt.Println("Enter the target word:")
	targetWord, _ := reader.ReadString('\n')
	targetWord = strings.TrimSpace(targetWord)

	synonym, ok := synonyms[targetWord]
	if ok {
		fmt.Println(synonym)
	}
}
