package cmd

import (
	"os"

	internal "github.com/GuryevVladislav/digital-showcase/golang/test_cases/internal"
	"github.com/spf13/cobra"
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "test_cases",
	Short: "A brief description of your application",
	Long: `A longer description that spans multiple lines and likely contains
examples and usage of using your application. For example:

Cobra is a CLI library for Go that empowers applications.
This application is a tool to generate the needed files
to quickly create a Cobra application.`,
}

var MapSynonymsCmd = &cobra.Command{
	Use:   "map-synonyms",
	Short: "",
	Long:  "",
	Run:   internal.MapSynonyms,
}

var NearestNumberCmd = &cobra.Command{
	Use:   "nearest-number",
	Short: "",
	Long:  "",
	Run:   internal.NearestNumber,
}

var PolyglotsCmd = &cobra.Command{
	Use:   "polyglots",
	Short: "",
	Long:  "",
	Run:   internal.Polyglots,
}

func Execute() {
	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}
}

func init() {
	rootCmd.AddCommand(
		MapSynonymsCmd,
		NearestNumberCmd,
		PolyglotsCmd,
	)
}
