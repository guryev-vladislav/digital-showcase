package commands

import (
	"bufio"
	"fmt"
	"os"
	"strconv"

	solutions "github.com/guryev-vladislav/digital-showcase/golang/test_cases/internal/solutions"
	"github.com/spf13/cobra"
)

// phoneNumbersCmd represents the phoneNumbers command
var PhoneNumbersCmd = &cobra.Command{
	Use:   "phoneNumbers",
	Short: "A brief description of your command",
	Long: `A longer description that spans multiple lines and likely contains examples
and usage of using your command. For example:

Cobra is a CLI library for Go that empowers applications.
This application is a tool to generate the needed files
to quickly create a Cobra application.`,
	Run: PhoneNumberCmd,
}

func init() {

}

func PhoneNumberCmd(cmd *cobra.Command, args []string) {
	fmt.Println("Enter the number of digits:")
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	_, err := strconv.Atoi(scanner.Text())
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	scanner.Scan()
	input := scanner.Text()

	ok := solutions.PhoneNumberCurrent(input)
	if ok {
		fmt.Println("1")
	} else {
		fmt.Println("0")
	}

}
