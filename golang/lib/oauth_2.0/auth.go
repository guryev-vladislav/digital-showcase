package oauth2

import (
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"net/url"
	"strings"
)

type AuthCodeResult struct {
	Code  string
	State string
	Error string
}

func GenerateState() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("failed to generate state: %w", err)
	}
	return base64.URLEncoding.EncodeToString(b), nil
}

func BuildAuthURL(cfg Config) (string, error) {
	if err := cfg.Validate(); err != nil {
		return "", err
	}

	state := cfg.State
	if state == "" {
		var err error
		state, err = GenerateState()
		if err != nil {
			return "", err
		}
	}

	authURL, err := url.Parse(cfg.AuthURL)
	if err != nil {
		return "", fmt.Errorf("invalid AuthURL: %w", err)
	}

	params := url.Values{}
	existingParams := authURL.Query()

	for key, values := range existingParams {
		for _, value := range values {
			params.Add(key, value)
		}
	}

	params.Set("response_type", "code")
	params.Set("client_id", cfg.ClientID)
	params.Set("redirect_uri", cfg.RedirectURI)
	params.Set("state", state)

	if len(cfg.Scopes) > 0 {
		formattedScope, err := formatScope(cfg.Scopes)
		if err != nil {
			return "", fmt.Errorf("invalid scope: %w", err)
		}
		params.Set("scope", formattedScope)
	}

	authURL.RawQuery = params.Encode()
	return authURL.String(), nil
}

func formatScope(scopes []string) (string, error) {
	if len(scopes) == 0 {
		return "", nil
	}

	for _, scope := range scopes {
		if scope == "" {
			return "", fmt.Errorf("scope cannot be empty")
		}
	}

	return strings.Join(scopes, " "), nil
}

func ParseAuthCodeCallback(redirectURL string, expectedState string) (*AuthCodeResult, error) {
	if redirectURL == "" {
		return nil, fmt.Errorf("redirectURL cannot be empty")
	}

	parsed, err := url.Parse(redirectURL)
	if err != nil {
		return nil, fmt.Errorf("invalid redirect URL: %w", err)
	}

	query := parsed.Query()
	result := &AuthCodeResult{
		Code:  query.Get("code"),
		State: query.Get("state"),
		Error: query.Get("error"),
	}

	if expectedState != "" && result.State != expectedState {
		return nil, fmt.Errorf("state parameter mismatch: expected %s, got %s", expectedState, result.State)
	}

	if result.Error != "" {
		return result, fmt.Errorf("authorization server returned error: %s", result.Error)
	}

	if result.Code == "" {
		return nil, fmt.Errorf("authorization code not found in response")
	}

	return result, nil
}
