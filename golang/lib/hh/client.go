package hh

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

type HHClient struct {
	baseURL           string
	client            *http.Client
	token             *Token
	clientID          string
	clientSecret      string
	redirectURI       string
	authorizationCode string
}

func NewClient(
	ctx context.Context,
	client *http.Client,
	baseURL, clientID, clientSecret, redirectURI, authorizationCode string,
) (*HHClient, error) {

	if client == nil {
		client = http.DefaultClient
	}

	clientHH := &HHClient{
		baseURL:           baseURL,
		client:            client,
		clientID:          clientID,
		clientSecret:      clientSecret,
		redirectURI:       redirectURI,
		authorizationCode: authorizationCode,
	}

	err := clientHH.getToken(ctx)
	if err != nil {
		return nil, err
	}

	go clientHH.updatePeriodToken(ctx)

	return clientHH, nil
}

func (c *HHClient) getToken(ctx context.Context) error {
	endpoint := c.baseURL + endpointCreateToken

	urlToken, err := url.Parse(endpoint)
	if err != nil {
		return fmt.Errorf("invalid AuthURL: %w", err)
	}

	params := url.Values{}
	existingParams := urlToken.Query()

	for key, values := range existingParams {
		for _, value := range values {
			params.Add(key, value)
		}
	}

	params.Set("grant_type", "authorization_code")
	params.Set("client_id", c.clientID)
	params.Set("client_secret", c.clientSecret)
	params.Set("redirect_uri", c.redirectURI)
	params.Set("code", c.authorizationCode)

	urlToken.RawQuery = params.Encode()

	body, err := c.post(ctx, urlToken.String(), nil)
	if err != nil {
		return err
	}

	token := Token{}
	err = json.Unmarshal(body, &token)
	if err != nil {
		return err
	}

	c.token = &token

	return nil
}

func (c *HHClient) refreshToken(ctx context.Context) error {
	endpoint := endpointCreateToken

	queryParams := url.Values{}
	queryParams.Add("grant_type", "refresh_token")
	queryParams.Add("refresh_token", c.token.RefreshToken)

	endpoint += "?" + queryParams.Encode()

	body, err := c.get(ctx, endpoint)
	if err != nil {
		return err
	}

	token := Token{}
	err = json.Unmarshal(body, &token)
	if err != nil {
		return err
	}

	c.token = &token

	return nil
}

func (c *HHClient) updatePeriodToken(ctx context.Context) {

	for {
		err := c.refreshToken(ctx)
		if err != nil {
			time.Sleep(250 * time.Millisecond)
			continue
		}
		time.Sleep(time.Duration(c.token.ExpiresIn/2) * time.Second)
	}
}

func (hc *HHClient) sendRequest(ctx context.Context, url, method string, body any) ([]byte, error) {

	var requestBody io.Reader
	if body != nil && method != http.MethodGet {
		requestBodyBytes, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}

		requestBody = bytes.NewBuffer(requestBodyBytes)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, requestBody)
	if err != nil {
		return nil, err
	}

	if hc.token != nil {
		req.Header.Set("Authorization", "Bearer "+hc.token.AccessToken)
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := hc.client.Do(req)
	if err != nil {
		return nil, err
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		err := fmt.Errorf("status Code: %v, status: %s", resp.StatusCode, resp.Status)
		return nil, err
	}

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	return responseBody, nil
}

func (hc *HHClient) get(ctx context.Context, endpoint string) ([]byte, error) {
	return hc.sendRequest(ctx, endpoint, http.MethodGet, nil)
}

func (hc *HHClient) post(ctx context.Context, endpoint string, body any) ([]byte, error) {
	return hc.sendRequest(ctx, endpoint, http.MethodPost, body)
}
