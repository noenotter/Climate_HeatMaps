import time
from pyngrok import ngrok

# --- 1) set your ngrok account token ---
ngrok.set_auth_token("32SxdtavCvO9FxP3m0eJTCH9Jjz_7Y4KM5muhaQ7Z7xF5Ut8r")

# --- 2) open tunnel to Streamlit on port 8501 with basic auth ---
tunnel = ngrok.connect(8501, "http", auth="client:SuperSecret!")

print("✅ Tunnel is live!")
print("🔗 Public URL:", tunnel.public_url)
print("🔑 Username: client")
print("🔑 Password: SuperSecret!")
print("\nKeep this script running (Ctrl+C to stop).")

# --- 3) keep alive until you stop it ---
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    ngrok.disconnect(tunnel.public_url)
    ngrok.kill()
    print("\nTunnel closed.")
