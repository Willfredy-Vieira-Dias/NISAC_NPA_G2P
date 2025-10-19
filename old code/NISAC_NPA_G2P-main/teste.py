import requests
r = requests.post(
  "https://nisac-npa-g2p.onrender.com/api/analyze",
  json={"query":"Vou fazer um desafio de quem fica mais tempo com roupas de pele pesadas que aquecem até o osso no Dubai em Outubro dia 12."},
  timeout=180
)
print(r.status_code, r.json())
