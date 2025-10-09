package nearestnumber

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
)

var NearestNumberCmd = &cobra.Command{
	Use:   "nearest_number",
	Short: "Найти ближайший элемент в массиве к заданному числу",
	Long: `Команда nearest_number находит в массиве элемент, самый близкий по величине к данному числу.

	Формат ввода:
	- Первая строка: размер массива N (натуральное число, не превышающее 1000)
	- Вторая строка: N целых чисел через пробел (числа по модулю не превышают 1000)
	- Третья строка: число x для поиска (целое число, по модулю не превышает 1000)

	Формат вывода:
	- Значение элемента массива, ближайшее к x
	- Если ближайших элементов несколько, выводится любой из них

	Пример использования:
	echo -e "5\n1 2 3 4 5\n3" | go run main.go nearest_number`,
	Run: nearestNumberCmdRun,
}

func nearestNumberCmdRun(cmd *cobra.Command, args []string) {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	n, _ := strconv.Atoi(scanner.Text())

	scanner.Scan()
	elementsStr := strings.Split(scanner.Text(), " ")
	elements := make([]int, n)
	for i, s := range elementsStr {
		elements[i], _ = strconv.Atoi(s)
	}

	scanner.Scan()
	x, _ := strconv.Atoi(scanner.Text())

	closest := NearestNumber(elements, x)

	fmt.Println(closest)
}
