package signalstations

import (
	"bufio"
	"fmt"
	"os"
	"strconv"

	"github.com/spf13/cobra"
)

// SignalStationsCmd represents the signalStations command
var SignalStationsCmd = &cobra.Command{
	Use:   "signalStations",
	Short: "A brief description of your command",
	Long: `A longer description that spans multiple lines and likely contains examples
and usage of using your command. For example:

Cobra is a CLI library for Go that empowers applications.
This application is a tool to generate the needed files
to quickly create a Cobra application.`,
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
