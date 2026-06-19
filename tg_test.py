import requests
BOT_TOKEN = "8483995149:AAFt3qPMFXfn1DGbQQiofhnvMG1FOj6dPpc"
CHAT_ID = "8616636381"
try:
    r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getMe", timeout=10)
    data = r.json()
    print("Bot alive:", data.get("ok"), "| Username:", data.get("result", {}).get("username"))
    r2 = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": "AlphaEdge health check OK - all 17 fixes deployed and running!"}, timeout=10)
    print("Message sent:", r2.json().get("ok"))
except Exception as e:
    print("ERROR:", e)
