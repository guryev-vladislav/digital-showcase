package nearestnumber

import (
	"math"
)

/*
Напишите программу, которая находит в массиве элемент, самый близкий по величине к данному числу.

Формат ввода
В первой строке задается одно натуральное число N, не превосходящее 1000 — размер массива.
Во второй строке содержатся N чисел — элементы массива, целые числа, не превосходящие по модулю 1000.
В третьей строке вводится одно целое число x, не превосходящее по модулю 1000.

Формат вывода
Вывести значение элемента массива, ближайшее к x. Если таких чисел несколько, выведите любое из них.
*/

func NearestNumber(elements []int, x int) int {

	closest := elements[0]
	minDiff := math.Abs(float64(elements[0] - x))
	for _, element := range elements {
		diff := math.Abs(float64(element - x))
		if diff < minDiff {
			minDiff = diff
			closest = element
		}
	}

	return closest

}
