import requests

token   = '8483995149:AAFm4c9eRSPPb7Fj9A2_vyyXDhTEEPRx89s'
chat_id = '8616636381'

url     = f'https://api.telegram.org/bot{token}/sendMessage'
payload = {
    'chat_id': chat_id,
    'text'   : 'AlphaEdge test message - New token working!'
}

r = requests.post(url, json=payload, timeout=10)
print('Status:', r.status_code)
print('Response:', r.text)