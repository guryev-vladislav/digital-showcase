import requests
import os

headers = {
    'accept': 'application/json',
    'accept-language': 'ru,en;q=0.9',
    'content-type': 'multipart/form-data; boundary=----WebKitFormBoundarynYPdLgg5k0tZaBGR',
    'dnt': '1',
    'origin': 'https://nn.hh.ru',
    'priority': 'u=1, i',
    'referer': 'https://nn.hh.ru/account/login?backurl=%2F&hhtmFrom=main',
    'sec-ch-ua': '"Chromium";v="130", "YaBrowser";v="24.12", "Not?A_Brand";v="99", "Yowser";v="2.5"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 YaBrowser/24.12.0.0 Safari/537.36',
    'x-gib-fgsscgib-w-hh': 'jwHV5cc6f6ed1d478d67bcd8c8d760933cd9d785',
    'x-gib-gsscgib-w-hh': 'edvS362OTuS0lFQyR6Dl8+OEXvK19lhfJej8b1/LVI6L1eHuabKStU9is8NdQ8zD+JzBQxn9UPfnk3ZjWDBfiwA07T5O/FSN/CkNhL1kKeaSYKA860vFn5SvMP6eQIAxFPLOMaKXww6yQ4OumEDOWrJjzb9ZvGTo9uiuX2Tqo3Ki0waO52JeEUUJBP/Iznm4VbQv7SjBkq6wbXO0jH/Y8HacZGKn/BKRG668KNgv9JoA9tTpv7eroFKkf2IQXA==',
    'x-hhtmfrom': 'main',
    'x-hhtmsource': 'account_login',
    'x-requested-with': 'XMLHttpRequest',
    'x-xsrftoken': '7d5485df54f21144a62d045da9d55501',
}

files = [
    ('_xsrf', (None, '7d5485df54f21144a62d045da9d55501')),
    ('failUrl', (None, '/account/login?backurl=%2F')),
    ('accountType', (None, 'APPLICANT')),
    ('remember', (None, 'yes')),
    ('username', (None, os.environ.get("USERNAME"))),
    ('password', (None, os.environ.get("PASSWORD"))),
    ('username', (None, os.environ.get("USERNAME"))),
    ('password', (None, os.environ.get("PASSWORD"))),
    ('isBot', (None, 'false')),
    ('captchaText', (None, '')),
]

response = requests.post('https://nn.hh.ru/account/login', params={'backurl': '/',}, headers=headers, files=files)

print(response.text)