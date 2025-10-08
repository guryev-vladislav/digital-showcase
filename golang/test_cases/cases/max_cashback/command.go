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
	Short: "A brief description of your command",
	Long: `A longer description that spans multiple lines and likely contains examples
and usage of using your command. For example:

Cobra is a CLI library for Go that empowers applications.
This application is a tool to generate the needed files
to quickly create a Cobra application.`,
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
