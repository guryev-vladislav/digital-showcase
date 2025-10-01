package oauth2

import (
	"fmt"
	"net/url"
	"time"
)

func GetAuthorizationCode(cfg Config) (*AuthCodeResult, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	// Проверяем, что RedirectURI - это localhost
	parsed, err := url.Parse(cfg.RedirectURI)
	if err != nil {
		return nil, fmt.Errorf("invalid RedirectURI: %w", err)
	}

	if parsed.Hostname() != "localhost" && parsed.Hostname() != "127.0.0.1" {
		return nil, fmt.Errorf("RedirectURI must be localhost for automatic callback handling")
	}

	// Генерируем state если не указан
	state := cfg.State
	if state == "" {
		var err error
		state, err = GenerateState()
		if err != nil {
			return nil, err
		}
		cfg.State = state
	}

	// Извлекаем порт из RedirectURI
	var port int
	if parsed.Port() != "" {
		fmt.Sscanf(parsed.Port(), "%d", &port)
	} else {
		port = 80 // default port для HTTP
	}

	// ИЗМЕНЕНО: извлекаем путь из RedirectURI
	path := parsed.Path
	if path == "" {
		path = "/"
	}

	// ИЗМЕНЕНО: передаем путь в сервер
	server := NewCallbackServer(path, state, 5*time.Minute)
	_, err = server.Start(port)
	if err != nil {
		return nil, err
	}
	defer server.Stop()

	// Генерируем Auth URL
	authURL, err := BuildAuthURL(cfg)
	if err != nil {
		return nil, err
	}

	fmt.Printf("1. Open URL in browser:\n%s\n\n", authURL)
	fmt.Printf("2. Waiting for callback on: %s\n\n", cfg.RedirectURI)

	// Ждем результат
	result, err := server.WaitForResult()
	if err != nil {
		return nil, fmt.Errorf("authorization timeout: %w", err)
	}

	if result.Error != "" {
		return nil, fmt.Errorf("authorization failed: %s", result.Error)
	}

	fmt.Printf("✅ Authorization code received: %s\n", result.Code)
	fmt.Printf("✅ State verified: %s\n", result.State)

	return result, nil
}
