package cases

import (
	"github.com/spf13/cobra"

	mapSynonyms "github.com/guryev-vladislav/digital-showcase/golang/test_cases/cases/map_synonyms"
	maxcashback "github.com/guryev-vladislav/digital-showcase/golang/test_cases/cases/max_cashback"
	nearestNumber "github.com/guryev-vladislav/digital-showcase/golang/test_cases/cases/nearest_number"
	phoneNumbers "github.com/guryev-vladislav/digital-showcase/golang/test_cases/cases/phone_numbers"
	polyglots "github.com/guryev-vladislav/digital-showcase/golang/test_cases/cases/polyglots"
	sygnalstations "github.com/guryev-vladislav/digital-showcase/golang/test_cases/cases/signal_stations"
)

func Register() []*cobra.Command {
	return []*cobra.Command{
		phoneNumbers.PhoneNumbersCmd,
		mapSynonyms.MapSynonymsCmd,
		nearestNumber.NearestNumberCmd,
		polyglots.PolyglotsCmd,
		maxcashback.MaxCashbackCmd,
		sygnalstations.SignalStationsCmd,
	}
}
