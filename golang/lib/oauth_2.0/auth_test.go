package oauth2_test

import (
	"fmt"
	"log"
	"testing"

	oauth2 "github.com/guryev-vladislav/digital-showcase/golang/lib/oauth_2.0"
)

func TestGenerateState(t *testing.T) {
	cfg := oauth2.Config{
		AuthURL:     "https://hh.ru/oauth/authorize",
		ClientID:    "V7O98HN4VUFAPU1PT08AT8U1GD5CM7L9F2JQSFRD4GGDRTS21CNQU92EGJVRU4U6",
		RedirectURI: "http://localhost:8080/hh/auth/callback", // ← путь /hh/auth/callback
		Scopes:      []string{},
	}

	result, err := oauth2.GetAuthorizationCode(cfg)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Use this code to get token: %s\n", result.Code)
}
