# test_webhook.py

"""
Quick test: creates a public URL for your webhook.
Use this URL in TradingView alerts.
"""

from pyngrok import ngrok
import time

print("\nStarting webhook tunnel...")

# Create public URL pointing to your local webhook
tunnel = ngrok.connect(5000)

print(f"\n{'='*60}")
print(f"YOUR PUBLIC WEBHOOK URL:")
print(f"{'='*60}")
print(f"\n   {tunnel.public_url}/webhook")
print(f"\n   Use this URL in TradingView alerts!")
print(f"\n   Health check: {tunnel.public_url}/health")
print(f"{'='*60}")
print("\nPress Ctrl+C to stop\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    ngrok.disconnect(tunnel.public_url)
    print("\nTunnel closed.")