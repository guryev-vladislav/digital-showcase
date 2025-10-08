package cmd

import (
	internal "github.com/guryev-vladislav/digital-showcase/tree/main/golang/test_cases/internal"

	"github.com/spf13/cobra"
)

// phoneNumbersCmd represents the phoneNumbers command
var phoneNumbersCmd = &cobra.Command{
	Use:   "phoneNumbers",
	Short: "A brief description of your command",
	Long: `A longer description that spans multiple lines and likely contains examples
and usage of using your command. For example:

Cobra is a CLI library for Go that empowers applications.
This application is a tool to generate the needed files
to quickly create a Cobra application.`,
	Run: internal.PhoneNumberCmd,
}

func init() {
	rootCmd.AddCommand(phoneNumbersCmd)

	// Here you will define your flags and configuration settings.

	// Cobra supports Persistent Flags which will work for this command
	// and all subcommands, e.g.:
	// phoneNumbersCmd.PersistentFlags().String("foo", "", "A help for foo")

	// Cobra supports local flags which will only run when this command
	// is called directly, e.g.:
	// phoneNumbersCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")
}
