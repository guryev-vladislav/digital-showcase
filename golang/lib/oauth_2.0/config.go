package oauth2

import (
	"fmt"
	"net/url"
)

type Config struct {
	AuthURL     string   `json:"auth_url"`
	ClientID    string   `json:"client_id"`
	RedirectURI string   `json:"redirect_uri"`
	Scopes      []string `json:"scopes"`
	State       string   `json:"state"`
}

func (c *Config) Validate() error {
	if c.AuthURL == "" {
		return fmt.Errorf("AuthURL is required")
	}
	if c.ClientID == "" {
		return fmt.Errorf("ClientID is required")
	}
	if c.RedirectURI == "" {
		return fmt.Errorf("RedirectURI is required")
	}

	if _, err := url.Parse(c.AuthURL); err != nil {
		return fmt.Errorf("invalid AuthURL: %w", err)
	}
	if _, err := url.Parse(c.RedirectURI); err != nil {
		return fmt.Errorf("invalid RedirectURI: %w", err)
	}

	for _, scope := range c.Scopes {
		if scope == "" {
			return fmt.Errorf("scope cannot be empty")
		}
	}

	return nil
}
