package polyglots

import (
	"bufio"
	"fmt"
	"os"
	"strconv"

	"github.com/spf13/cobra"
)

// PolyglotsCmd represents the polyglots command
var PolyglotsCmd = &cobra.Command{
	Use: "polyglots",
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
