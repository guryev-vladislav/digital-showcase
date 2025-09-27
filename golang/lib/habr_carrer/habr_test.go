package habr_carrer

import (
	"context"
	"testing"
)

func TestHabrClient_Authorization(t *testing.T) {
	ctx := context.Background()
	host := ""
	clientID := ""
	clientSecret := ""
	redirectURI := ""

	_, err := NewHabrClient(ctx, nil, host, clientID, clientSecret, redirectURI)
	if err != nil {
		t.Fatal(err)
	}
}
