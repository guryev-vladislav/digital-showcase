package mapsynonyms

import "strings"

func MapSynonyms(lines []string, targetWord string) (string, bool) {
	synonyms := make(map[string]string, len(lines)*2)
	for _, line := range lines {
		words := strings.Split(strings.TrimSpace(line), " ")
		word1, word2 := words[0], words[1]
		synonyms[word1] = word2
		synonyms[word2] = word1
	}

	synonym, ok := synonyms[targetWord]
	if ok {
		return synonym, true
	}

	return "", false
}
