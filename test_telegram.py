# test_telegram.py - Advanced Diagnostic Version
import os
import socket
import requests
from dotenv import load_dotenv

load_dotenv('config/secrets.env')

token = os.getenv('TELEGRAM_BOT_TOKEN', '')
chat_id = os.getenv('TELEGRAM_CHAT_ID', '')

print("=" * 60)
print("ALPHA EDGE - TELEGRAM DEEP DIAGNOSTIC")
print("=" * 60)

# ============================================================
# TEST 1 - NETWORK CHECKS
# ============================================================
print("\n📋 TEST 1: Network Checks")
print("-" * 60)

def check_host(host, port):
    try:
        socket.setdefaulttimeout(5)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.close()
        return True
    except Exception as e:
        return False

print(f"  Google DNS (8.8.8.8:53)          : {'✅' if check_host('8.8.8.8', 53) else '❌'}")
print(f"  Telegram API port 443            : {'✅' if check_host('api.telegram.org', 443) else '❌'}")
print(f"  Telegram API port 80             : {'✅' if check_host('api.telegram.org', 80) else '❌'}")

# ============================================================
# TEST 2 - GET BOT INFO
# ============================================================
print("\n📋 TEST 2: Bot Token Validation")
print("-" * 60)

try:
    url = f"https://api.telegram.org/bot{token}/getMe"
    r = requests.get(url, timeout=15)
    if r.ok:
        info = r.json().get('result', {})
        print(f"  ✅ Bot Valid: @{info.get('username')}")
    else:
        print(f"  ❌ Bot Invalid: {r.text}")
except Exception as e:
    print(f"  ❌ getMe Error: {type(e).__name__}: {e}")

# ============================================================
# TEST 3 - GET UPDATES (Find Real Chat ID)
# ============================================================
print("\n📋 TEST 3: Get Updates - Find Your Real Chat ID")
print("-" * 60)

try:
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    r = requests.get(url, timeout=15)
    print(f"  Status Code: {r.status_code}")

    if r.ok:
        data = r.json()
        updates = data.get('result', [])
        if updates:
            print(f"  ✅ Found {len(updates)} message(s)!")
            print()
            for u in updates:
                msg = u.get('message', {})
                chat = msg.get('chat', {})
                user = msg.get('from', {})
                real_chat_id = chat.get('id')
                print(f"  ✅ REAL CHAT ID  : {real_chat_id}")
                print(f"     From         : {user.get('first_name')}")
                print(f"     Message      : {msg.get('text')}")
                print()
            print("  ⬆️  Copy the REAL CHAT ID above")
            print("  ⬆️  Update TELEGRAM_CHAT_ID in secrets.env")
        else:
            print("  ⚠️  No updates found yet")
            print("  Go to Telegram → @Alphaedge_blebu_bot")
            print("  Send another message then run this again")
    else:
        print(f"  ❌ getUpdates failed: {r.text}")

except Exception as e:
    print(f"  ❌ getUpdates Error: {type(e).__name__}: {e}")

# ============================================================
# TEST 4 - SEND MESSAGE WITH DETAILED ERROR
# ============================================================
print("\n📋 TEST 4: Send Message - Detailed Error Report")
print("-" * 60)

print(f"  Using Chat ID: {chat_id}")
print(f"  Using Token  : {token[:10]}...{token[-6:]}")
print()

try:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    print(f"  Calling URL: {url[:50]}...")
    print()

    r = requests.post(
        url,
        json={
            'chat_id': chat_id,
            'text': '✅ Alpha Edge Test'
        },
        timeout=30
    )

    print(f"  HTTP Status : {r.status_code}")
    print(f"  Response    : {r.text}")

    if r.ok:
        print()
        print("  ✅ MESSAGE SENT SUCCESSFULLY!")
        print("  Check your Telegram app now!")
    else:
        print()
        print("  ❌ Message failed - see response above")

except requests.exceptions.SSLError as e:
    print(f"  ❌ SSL Error: {e}")
    print("  Fix: pip install --upgrade certifi")

except requests.exceptions.ProxyError as e:
    print(f"  ❌ Proxy Error: {e}")
    print("  Fix: Disable proxy or VPN")

except requests.exceptions.ConnectionError as e:
    print(f"  ❌ Connection Error Details:")
    print(f"     Type    : {type(e).__name__}")
    print(f"     Message : {e}")
    print()
    print("  POSSIBLE CAUSES:")
    print("  1. Antivirus blocking POST requests")
    print("  2. Windows Firewall blocking Python")
    print("  3. ISP blocking Telegram sendMessage")
    print("  4. SSL certificate issue")
    print()
    print("  THINGS TO TRY:")
    print("  A. Run CMD as Administrator")
    print("  B. Temporarily disable antivirus")
    print("  C. Try mobile hotspot instead of WiFi")

except requests.exceptions.Timeout:
    print("  ❌ Timeout - Request took too long")
    print("  Try running again")

except Exception as e:
    print(f"  ❌ Unexpected Error:")
    print(f"     Type    : {type(e).__name__}")
    print(f"     Message : {e}")

# ============================================================
# TEST 5 - TRY WITH REAL CHAT ID FROM UPDATES
# ============================================================
print("\n📋 TEST 5: Try Sending With Chat ID From Updates")
print("-" * 60)

try:
    url_updates = f"https://api.telegram.org/bot{token}/getUpdates"
    r = requests.get(url_updates, timeout=15)

    if r.ok:
        updates = r.json().get('result', [])
        if updates:
            real_id = updates[-1]['message']['chat']['id']
            print(f"  Real Chat ID from updates: {real_id}")
            print(f"  Your secrets.env Chat ID : {chat_id}")

            if str(real_id) != str(chat_id):
                print()
                print("  ⚠️  CHAT ID MISMATCH DETECTED!")
                print(f"  secrets.env has : {chat_id}")
                print(f"  Should be       : {real_id}")
                print()
                print("  Fix: Update secrets.env with correct ID")
                print(f"  TELEGRAM_CHAT_ID={real_id}")

            print()
            print(f"  Trying to send to real ID: {real_id}")

            url_send = f"https://api.telegram.org/bot{token}/sendMessage"
            r2 = requests.post(
                url_send,
                json={
                    'chat_id': real_id,
                    'text': '✅ Alpha Edge Connected!'
                },
                timeout=30
            )
            print(f"  Status: {r2.status_code}")
            print(f"  Result: {r2.text}")

            if r2.ok:
                print()
                print("  ✅ SUCCESS! Check your Telegram!")
                print(f"  Update secrets.env:")
                print(f"  TELEGRAM_CHAT_ID={real_id}")
        else:
            print("  ⚠️  No messages in getUpdates")
            print("  Send a message to your bot first")
    else:
        print(f"  ❌ Could not get updates: {r.text}")

except Exception as e:
    print(f"  ❌ Error: {type(e).__name__}: {e}")

print()
print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

# ============================================================
# TEST 6 - TRY DIFFERENT CONNECTION METHODS
# ============================================================
print("\n📋 TEST 6: Alternative Connection Methods")
print("-" * 60)

import ssl
import urllib.request
import json

# Method A - Using GET instead of POST
print("  Method A: Using GET request instead of POST...")
try:
    import urllib.parse
    message = "Alpha Edge Test via GET"
    encoded = urllib.parse.quote(message)
    url = (
        f"https://api.telegram.org/bot{token}"
        f"/sendMessage?chat_id={chat_id}&text={encoded}"
    )
    req = urllib.request.urlopen(url, timeout=15)
    result = json.loads(req.read())
    if result.get('ok'):
        print("  ✅ SUCCESS via GET method!")
        print("  Check your Telegram now!")
    else:
        print(f"  ❌ Failed: {result}")
except Exception as e:
    print(f"  ❌ GET method failed: {type(e).__name__}: {e}")

# Method B - Using urllib instead of requests
print()
print("  Method B: Using urllib instead of requests...")
try:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({
        'chat_id': chat_id,
        'text': 'Alpha Edge Test via urllib'
    }).encode('utf-8')

    req = urllib.request.Request(
        url,
        data=data,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
        result = json.loads(resp.read())
        if result.get('ok'):
            print("  ✅ SUCCESS via urllib!")
            print("  Check your Telegram now!")
        else:
            print(f"  ❌ Failed: {result}")

except Exception as e:
    print(f"  ❌ urllib failed: {type(e).__name__}: {e}")

# Method C - Using httpx if available
print()
print("  Method C: Using httpx library...")
try:
    import httpx
    with httpx.Client(timeout=15) as client:
        r = client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                'chat_id': chat_id,
                'text': 'Alpha Edge Test via httpx'
            }
        )
        if r.status_code == 200:
            print("  ✅ SUCCESS via httpx!")
        else:
            print(f"  ❌ Failed: {r.text}")
except ImportError:
    print("  ⏭️  httpx not installed")
    print("  Install with: pip install httpx")
except Exception as e:
    print(f"  ❌ httpx failed: {type(e).__name__}: {e}")

print()
print("=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)