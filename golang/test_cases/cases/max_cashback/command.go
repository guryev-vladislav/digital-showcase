package maxcashback

import (
	"bufio"
	"fmt"
	"os"
	"strconv"

	"github.com/spf13/cobra"
)

// maxCashbackCmd represents the maxCashback command
var MaxCashbackCmd = &cobra.Command{
	Use:   "max_cashback",
	Short: "Рассчитать максимальный кешбэк за покупки",
	Long: `Команда max_cashback вычисляет максимальный кешбэк, который клиент может получить
за свои покупки согласно условиям программы лояльности.

Формат ввода:
- Первая строка: минимальная стоимость покупки (a)
- Вторая строка: сумма кешбэка за покупку (b) 
- Третья строка: общая сумма потраченных денег (x)

Пример использования:
echo -e "100\n5\n450" | go run main.go max_cashback`,
	Run: MaxCashbackCmdRun,
}

func MaxCashbackCmdRun(cmd *cobra.Command, args []string) {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	min, _ := strconv.Atoi(scanner.Text())
	scanner.Scan()
	cashback, _ := strconv.Atoi(scanner.Text())
	scanner.Scan()
	sum, _ := strconv.Atoi(scanner.Text())

	result := MaxCashBack(min, cashback, sum)
	fmt.Println(result)
}
