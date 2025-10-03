package habr_carrer

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
)

type HabrClient struct {
	client       *http.Client
	host         string
	accessToken  string
	clientID     string
	clientSecret string
	redirectURI  string
}

func NewHabrClient(
	ctx context.Context,
	httpClient *http.Client,
	host, clientID, clientSecret, redirectURI, authorizationCode string,
) (*HabrClient, error) {

	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	habrClient := &HabrClient{
		client:       httpClient,
		host:         host,
		clientID:     clientID,
		clientSecret: clientSecret,
		redirectURI:  redirectURI,
	}

	err := habrClient.createAccessToken(ctx, clientID, clientSecret, redirectURI, authorizationCode)
	if err != nil {
		return nil, err
	}

	return habrClient, nil
}

func (hc *HabrClient) createAccessToken(
	ctx context.Context,
	clientID, clientSecret, redirectURI, authorizationCode string,
) error {
	endpoint := fmt.Sprintf(endpointCreateToken, clientID, clientSecret, redirectURI, authorizationCode)

	body, err := hc.get(ctx, endpoint)
	if err != nil {
		return err
	}

	token := Token{}
	err = json.Unmarshal(body, &token)
	if err != nil {
		return err
	}

	hc.accessToken = token.AccessToken

	return nil
}

func (hc *HabrClient) GetUser(ctx context.Context) (*User, error) {
	endpoint := endpointGetUser
	body, err := hc.get(ctx, endpoint)
	if err != nil {
		return nil, err
	}

	user := User{}
	err = json.Unmarshal(body, &user)
	if err != nil {
		return nil, err
	}
	return &user, nil
}

func (hc *HabrClient) GetVacancies(ctx context.Context, page int) (*Vacancies, error) {
	endpoint := endpointGetVacancies
	body, err := hc.get(ctx, endpoint)
	if err != nil {
		return nil, err
	}

	queryParams := url.Values{}
	if page > 0 {
		queryParams.Add("page", strconv.Itoa(page))
	}
	if len(queryParams) > 0 {
		endpoint += "?" + queryParams.Encode()
	}

	vacancyResponse := Vacancies{}
	err = json.Unmarshal(body, &vacancyResponse)
	if err != nil {
		return nil, err
	}
	return &vacancyResponse, nil

}

func (hc *HabrClient) GetArchivedVacancies(ctx context.Context, page int) (*Vacancies, error) {
	endpoint := endpointGetArchivedVacancies
	body, err := hc.get(ctx, endpoint)
	if err != nil {
		return nil, err
	}

	queryParams := url.Values{}
	if page > 0 {
		queryParams.Add("page", strconv.Itoa(page))
	}
	if len(queryParams) > 0 {
		endpoint += "?" + queryParams.Encode()
	}

	vacancyResponse := Vacancies{}
	err = json.Unmarshal(body, &vacancyResponse)
	if err != nil {
		return nil, err
	}
	return &vacancyResponse, nil
}

func (hc *HabrClient) GetVacancy(ctx context.Context, id int) (*Vacancy, error) {
	endpoint := fmt.Sprintf(endpointGetVacancy, id)
	body, err := hc.get(ctx, endpoint)
	if err != nil {
		return nil, err
	}

	vacancy := Vacancy{}
	err = json.Unmarshal(body, &vacancy)
	if err != nil {
		return nil, err
	}
	return &vacancy, nil
}

func (hc *HabrClient) GetCompanies(ctx context.Context) (*Companies, error) {
	endpoint := endpointGetCompanies
	body, err := hc.get(ctx, endpoint)
	if err != nil {
		return nil, err
	}

	companies := Companies{}
	err = json.Unmarshal(body, &companies)
	if err != nil {
		return nil, err
	}
	return &companies, nil
}

func (hc *HabrClient) GetSkills(ctx context.Context, params SkillsParams) (*Skills, error) {
	endpoint := endpointGetSkills

	queryParams := url.Values{}
	if params.Term != "" {
		queryParams.Add("term", params.Term)
	}
	if params.Page > 0 {
		queryParams.Add("page", strconv.Itoa(params.Page))
	}
	if params.PerPage > 0 {
		queryParams.Add("per_page", strconv.Itoa(params.PerPage))
	}

	if len(queryParams) > 0 {
		endpoint += "?" + queryParams.Encode()
	}

	body, err := hc.get(ctx, endpoint)
	if err != nil {
		return nil, err
	}

	skills := Skills{}
	err = json.Unmarshal(body, &skills)
	if err != nil {
		return nil, err
	}
	return &skills, nil
}

func (hc *HabrClient) sendRequest(ctx context.Context, endpoint, method string, body any) ([]byte, error) {

	url := hc.host + endpoint

	if hc.accessToken != "" {
		url += "?access_token=" + hc.accessToken
	}

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

func (hc *HabrClient) get(ctx context.Context, endpoint string) ([]byte, error) {
	return hc.sendRequest(ctx, endpoint, http.MethodGet, nil)
}

func (hc *HabrClient) post(ctx context.Context, endpoint string, body any) ([]byte, error) {
	return hc.sendRequest(ctx, endpoint, http.MethodPost, body)
}
