import requests
r = requests.post(
  "https://shadowlike-tawanda-edgily.ngrok-free.dev/api/analyze",
  json={"query":"Vou fazer um casamento em luanda 23 de junho"},
  timeout=180
)
print(r.status_code, r.json())
