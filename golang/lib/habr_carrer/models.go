package habr_carrer

// User представляет модель пользователя
type User struct {
	Login      string       `json:"login"`       // Логин пользователя
	Email      string       `json:"email"`       // Email пользователя
	FirstName  string       `json:"first_name"`  // Имя
	LastName   string       `json:"last_name"`   // Фамилия
	MiddleName string       `json:"middle_name"` // Отчество
	Birthday   string       `json:"birthday"`    // День рождения
	Avatar     string       `json:"avatar"`      // Ссылка на аватар пользователя
	Location   UserLocation `json:"location"`    // Информация о местоположении пользователя
	Gender     string       `json:"gender"`      // Пол
}

// UserLocation представляет информацию о местоположении пользователя
type UserLocation struct {
	City    string `json:"city"`    // Город
	Country string `json:"country"` // Страна
}

// Vacancies представляет ответ API со списком вакансий
type Vacancies struct {
	Vacancies  []Vacancy  `json:"vacancies"`  // Список вакансий
	Pagination Pagination `json:"pagination"` // Информация о пагинации
}

// Vacancy представляет модель вакансии
type Vacancy struct {
	ID              int               `json:"id"`              // ID вакансии
	Title           string            `json:"title"`           // Заголовок вакансии
	Divisions       []Division        `json:"divisions"`       // Подразделения
	Specializations []Specialization  `json:"specializations"` // Специализации
	PublishedAt     string            `json:"published_at"`    // Дата публикации
	URL             string            `json:"url"`             // Ссылка на вакансию
	Qualification   *Qualification    `json:"qualification"`   // Квалификация (может быть null)
	City            string            `json:"city"`            // Город размещения вакансии
	Marked          bool              `json:"marked"`          // Признак того, что вакансия выделена
	Company         Company           `json:"company"`         // Информация о компании
	EmploymentType  string            `json:"employment_type"` // Тип трудоустройства
	Salary          string            `json:"salary"`          // Вилка зарплат
	Remote          bool              `json:"remote"`          // Удаленная работа
	ExpandedSalary  ExpandedSalary    `json:"expanded_salary"` // Детальная информация о зарплате
	Published       bool              `json:"published"`       // Статус публикации
	Paid            bool              `json:"paid"`            // Статус оплаты
	Skills          []VacancySkill    `json:"skills"`          // Навыки
	Locations       []VacancyLocation `json:"locations"`       // Локации (может быть null)
	Description     string            `json:"description"`     // Описание вакансии
	Bonuses         string            `json:"bonuses"`         // Бонусы вакансии
	Instructions    string            `json:"instructions"`    // Дополнительные инструкции
	Team            string            `json:"team"`            // Описание вашей команды
	Candidate       string            `json:"candidate"`       // Ожидания от кандидата
}

// Specialization представляет специализацию вакансии
type Specialization struct {
	ID    int         `json:"id"`    // ID специализации
	Title TitleLocale `json:"title"` // Название специализации
}

// TitleLocale представляет локализованные названия
type TitleLocale struct {
	RU string `json:"ru"` // Название на русском языке
	EN string `json:"en"` // Название на англ. языке
}

// Qualification представляет квалификацию
type Qualification struct {
	Title TitleLocale `json:"title"` // Название квалификации
}

// Company представляет информацию о компании
type Company struct {
	Name      string `json:"name"`       // Название компании
	AliasName string `json:"alias_name"` // Алиас компании
	URL       string `json:"url"`        // Ссылка на компанию
	LogoURL   string `json:"logo_url"`   // Ссылка на логотип компании
}

// Companies представляет список компаний
type Companies struct {
	Companies []Company `json:"companies"` // Список компаний
}

// ExpandedSalary представляет детальную информацию о зарплате
type ExpandedSalary struct {
	From     *int   `json:"from"`     // Минимальная зарплата (может быть null)
	To       *int   `json:"to"`       // Максимальная зарплата (может быть null)
	Currency string `json:"currency"` // Валюта зарплаты
}

// VacancySkill представляет навык в вакансии
type VacancySkill struct {
	Value int    `json:"value"` // ID навыка
	Alias string `json:"alias"` // Алиас навыка
	Title string `json:"title"` // Название навыка
}

// Division представляет подразделение
type Division struct {
	// Структура будет дополнена при наличии данных
}

// VacancyLocation представляет локацию вакансии
type VacancyLocation struct {
	Title string `json:"title"` // Название локации
	Href  string `json:"href"`  // Ссылка на локацию
}

// ResponseList представляет ответ API со списком откликов
type ResponseList struct {
	Responses  []Response `json:"responses"`  // Список откликов
	Pagination Pagination `json:"pagination"` // Информация о пагинации
}

// Response представляет модель отклика на вакансию
type Response struct {
	ID        int          `json:"id"`         // ID отклика
	VacancyID int          `json:"vacancy_id"` // ID вакансии
	User      ResponseUser `json:"user"`       // Информация о пользователе, оставившем отклик
	Body      string       `json:"body"`       // Сопроводительное письмо
	Favorite  bool         `json:"favorite"`   // Признак того, что отклик отмечен пользователем, разместившим вакансию
	Archived  bool         `json:"archived"`   // Признак того, что отклик удален пользователем, разместившим вакансию
	CreatedAt string       `json:"created_at"` // Дата размещения отклика на вакансию
}

// ResponseUser представляет информацию о пользователе в отклике
type ResponseUser struct {
	Login           string         `json:"login"`            // Логин пользователя, оставившего отклик
	Name            string         `json:"name"`             // Имя пользователя
	Avatar          string         `json:"avatar"`           // Ссылка на аватар пользователя
	Birthday        string         `json:"birthday"`         // День рождения
	Specialization  string         `json:"specialization"`   // Специализация
	Skills          []Skill        `json:"skills"`           // Навыки пользователя
	ExperienceTotal UserExperience `json:"experience_total"` // Общий стаж пользователя
	Relocation      bool           `json:"relocation"`       // Показатель готовности пользователя к релокации
	Remote          bool           `json:"remote"`           // Показатель готовности пользователя к удаленной работе
	Compensation    Compensation   `json:"compensation"`     // Ожидаемое вознаграждение
	WorkState       string         `json:"work_state"`       // Статус готовности к работе
	Age             int            `json:"age"`              // Возраст
	Location        UserLocation   `json:"location"`         // Местоположение пользователя
	Experiences     ExperienceItem `json:"experiences"`      // Опыт работы
	Educations      Education      `json:"educations"`       // Образование
}

// Skill представляет навык пользователя
type Skill struct {
	Title     string `json:"title"`      // Название навыка пользователя
	AliasName string `json:"alias_name"` // Алиас навыка
}

// Skills представляет список навыков
type Skills struct {
	Skills []Skill `json:"skills"` // Список навыков
}

// SkillsParams представляет параметры для получения списка навыков
type SkillsParams struct {
	Term    string `url:"term,omitempty"`     // поисковый запрос
	Page    int    `url:"page,omitempty"`     // номер страницы
	PerPage int    `url:"per_page,omitempty"` // количество записей на страницу
}

// UserExperience представляет опыт работы пользователя
type UserExperience struct {
	Month int `json:"month"` // Общий стаж пользователя в месяцах
}

// Compensation представляет ожидаемое вознаграждение
type Compensation struct {
	Value    int    `json:"value"`    // Ожидаемое вознаграждение
	Currency string `json:"currency"` // Валюта
}

// ExperienceItem представляет опыт работы на последнем месте
type ExperienceItem struct {
	Company  string `json:"company"`  // Компания последнего места работы
	Position string `json:"position"` // Должность на последнем месте работы
	Period   string `json:"period"`   // Продолжительность работы на последнем месте
}

// Education представляет образование пользователя
type Education struct {
	University string `json:"university"` // Последнее место обучения
	Faculty    string `json:"faculty"`    // Факультет
	StartDate  string `json:"start_date"` // Начало обучения
	EndDate    string `json:"end_date"`   // Окончание обучения
}

// Pagination представляет информацию о пагинации
type Pagination struct {
	Total int `json:"total"` // Общее количество элементов
	Page  int `json:"page"`  // Текущая страница
	Per   int `json:"per"`   // Количество элементов на странице
}

type Token struct {
	AccessToken string `json:"access_token"`
	TokenType   string `json:"token_type"`
	Scope       string `json:"scope"`
	CreatedAt   int    `json:"created_at"`
}
