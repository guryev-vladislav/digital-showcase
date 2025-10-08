package internal

import (
	"bufio"
	"fmt"
	"os"
	"strconv"

	"github.com/spf13/cobra"
)

// Телефонные номера

// В России мобильные номера состоят из 11 цифр и начинаются с «79».
// Вам дана строка цифр. Верно ли, что ее можно полностью разрезать на непересекающиеся непрерывные блоки, которые могут быть корректными номерами российских мобильных телефонов?
// Под «полностью разрезать» подразумевается, что каждая цифра строки должна принадлежать ровно одному блоку.

// Формат входных данных
// Первая строка входных данных содержит одно целое число n — длину строки (1 ≤ n ≤ 104). Во второй строке расположена сама строка — n десятичных цифр от 0 до 9.

// Формат выходных данных
// Выведи 1, если строку можно полностью разрезать на непересекающиеся непрерывные блоки, которые могут быть корректными номерами российских мобильных телефонов, и 0 в противном случае.

func PhoneNumberCurrent(n int, input string) bool {

	return true
}

func PhoneNumberCmd(cmd *cobra.Command, args []string) {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	n, err := strconv.Atoi(scanner.Text())
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	scanner.Scan()
	input := scanner.Text()

	ok := PhoneNumberCurrent(n, input)
	if ok {
		fmt.Println("1")
	} else {
		fmt.Println("0")
	}

}
