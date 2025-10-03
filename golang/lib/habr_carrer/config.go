package habr_carrer

// client_id, client_secret, redirect_uri, authorization_code
const endpointCreateToken = "/integrations/oauth/token?client_id={%s}&client_secret={%s}&redirect_uri={%s}&grant_type=authorization_code&code={%s}"

// client_id, redirect_uri
const EndpointAuthorize = "/integrations/oauth/authorize?client_id={%s}&redirect_uri={%s}&response_type=code"

const endpointGetUser = "/v1/integrations/users/me"
const endpointGetVacancies = "/v1/integrations/vacancies"
const endpointGetArchivedVacancies = "/v1/integrations/vacancies/archived"
const endpointGetVacancy = "/v1/integrations/vacancies/:%d"
const endpointGetCompanies = "/v1/integrations/companies/my"
const endpointGetSkills = "/v1/integrations/skills"
