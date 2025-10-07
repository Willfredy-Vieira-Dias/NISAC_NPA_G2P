import requests
r = requests.post(
  "https://nisac-npa-g2p.onrender.com/api/analyze",
  json={"query":""},
  timeout=180
)
print(r.status_code, r.json())
