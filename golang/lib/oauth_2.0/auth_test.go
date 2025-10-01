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
		ClientID:    "",
		RedirectURI: "http://localhost:8080/hh/auth/callback", // ← путь /hh/auth/callback
		Scopes:      []string{},
	}

	result, err := oauth2.GetAuthorizationCode(cfg)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Use this code to get token: %s\n", result.Code)
}

func TestParseAuthCodeCallback(t *testing.T) {
	cfg := oauth2.Config{
		AuthURL:     "https://hh.ru/oauth/authorize",
		ClientID:    "your_client_id",
		RedirectURI: "http://localhost:8080/", // порт указывается здесь
		Scopes:      []string{},
	}

	result, err := oauth2.GetAuthorizationCode(cfg)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Use this code to get token: %s\n", result.Code)
}
