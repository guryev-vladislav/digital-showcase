package phonenumbers

import (
	"bufio"
	"fmt"
	"os"
	"strconv"

	"github.com/spf13/cobra"
)

var PhoneNumbersCmd = &cobra.Command{
	Use:   "phone_numbers",
	Short: "Проверить возможность разбиения строки на корректные российские номера телефонов",
	Long: `Команда phone_numbers проверяет, можно ли строку цифр полностью разрезать 
	на непересекающиеся блоки, которые являются корректными российскими номерами телефонов.

	В России мобильные номера состоят из 11 цифр и начинаются с «79».
	Каждая цифра строки должна принадлежать ровно одному блоку.

	Формат ввода:
	- Первая строка: длина строки n (1 ≤ n ≤ 10^4)
	- Вторая строка: строка из n десятичных цифр от 0 до 9

	Формат вывода:
	- 1, если строку можно разрезать на корректные номера телефонов
	- 0, если это невозможно

	Пример использования:
	echo -e "22\n7979797979797979797979" | go run main.go phone_numbers`,
	Run: PhoneNumberCmd,
}

func PhoneNumberCmd(cmd *cobra.Command, args []string) {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	_, err := strconv.Atoi(scanner.Text())
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	scanner.Scan()
	input := scanner.Text()

	ok := PhoneNumberCurrent(input)
	if ok {
		fmt.Println("1")
	} else {
		fmt.Println("0")
	}

}
