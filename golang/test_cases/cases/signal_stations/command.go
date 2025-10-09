package signalstations

import (
	"bufio"
	"fmt"
	"os"
	"strconv"

	"github.com/spf13/cobra"
)

var SignalStationsCmd = &cobra.Command{
	Use:   "signal_stations",
	Short: "Найти площадь треугольника, образованного базовыми станциями",
	Long: `Команда signal_stations вычисляет квадрат площади треугольника, образованного тремя базовыми станциями.

	На территории расположены три базовые станции с радиусами покрытия a, b и c.
	Сигнал каждой станции принимается на расстоянии радиуса покрытия или меньше от станции.
	Условия измерения сигнала:
	- есть ровно одна точка приема сигналов от станций 1 и 2
	- есть ровно одна точка приема сигналов от станций 2 и 3  
	- есть ровно одна точка приема сигналов от станций 1 и 3

	Формат ввода:
	- Первая строка: радиус покрытия первой станции a (1 ≤ a ≤ 100)
	- Вторая строка: радиус покрытия второй станции b (1 ≤ b ≤ 100)
	- Третья строка: радиус покрытия третьей станции c (1 ≤ c ≤ 100)

	Формат вывода:
	- Два целых числа: минимальное и максимальное значение квадрата площади треугольника
	- Если треугольник невозможен: -1

	Пример использования:
	echo -e "3\n4\n5" | go run main.go signal_stations`,
	Run: SignalStationsCmdRun,
}

func SignalStationsCmdRun(cmd *cobra.Command, args []string) {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	signal1, _ := strconv.Atoi(scanner.Text())
	scanner.Scan()
	signal2, _ := strconv.Atoi(scanner.Text())
	scanner.Scan()
	signal3, _ := strconv.Atoi(scanner.Text())

	area := SignalStations(signal1, signal2, signal3)
	if area == -1 {
		fmt.Println(area)
	} else {
		fmt.Println(area, area)
	}
}
