package hh

import (
	"context"
	"testing"
)

func TestAuthorization(t *testing.T) {

	baseURL := "https://api.hh.ru"
	clientID := "V7O98HN4VUFAPtjktyktU1PT08AT8U1FRD4GGDRTS21CNQU92EGJVRU4U6"
	clientSecret := "IQ3TDFD2Qrstynrthrt567NSKEF0JKJLDNJUSS4P2LS9O8LRLVQ6060S3E"
	redirectURI := "http://localhost:8080/hh/auth/callback"
	authorizationCode := "JQCPIOJA01VSL8674HQVG3UL4Q3CQHFRDANPC4U9PQS85V6TRS3FVOM6IKTBNNHP"

	client, err := NewClient(context.Background(), nil, baseURL, clientID, clientSecret, redirectURI, authorizationCode)
	if err != nil {
		t.Fatal(err)
	}

	if client == nil {
		t.Fatal("client is nil")
	}
}
