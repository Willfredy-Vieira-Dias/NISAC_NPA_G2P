import requests
import json

# URL da sua API no Azure
base_url = "https://cygnusx1api2025-e7b8ctf6e6bucpam.canadacentral-01.azurewebsites.net"

print("=" * 60)
print("TESTANDO API NO AZURE")
print("=" * 60)

# Teste 1: Verificar se a API está rodando
print("\n1. Testando endpoint raiz (GET /)...")
try:
    response = requests.get(f"{base_url}/")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Resposta: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"   Erro: {response.text}")
except Exception as e:
    print(f"   ERRO: {e}")

# Teste 2: Endpoint de teste
print("\n2. Testando endpoint /test...")
try:
    response = requests.get(f"{base_url}/test")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Resposta: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"   Erro: {response.text}")
except Exception as e:
    print(f"   ERRO: {e}")

# Teste 3: Endpoint principal com POST
print("\n3. Testando endpoint /api/analyze (POST)...")
test_data = {
    "query": "Vou fazer um churrasco em Luanda próximo fim de semana"
}

try:
    response = requests.post(
        f"{base_url}/api/analyze",
        json=test_data,
        headers={"Content-Type": "application/json"},
        timeout=60  # 60 segundos de timeout
    )
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("   ✅ SUCESSO! API funcionando corretamente")
        print(f"   Resumo rápido:")
        if "quick_summary" in result:
            print(f"     - Probabilidade de chuva: {result['quick_summary'].get('rain_probability', 'N/A')}%")
            print(f"     - Risco de calor: {result['quick_summary'].get('heat_risk', 'N/A')}%")
            print(f"     - Nível de risco: {result['quick_summary'].get('risk_level', 'N/A')}")
    else:
        print(f"   ❌ Erro: {response.text}")
except requests.exceptions.Timeout:
    print("   ⏱️ TIMEOUT: A requisição demorou muito (pode ser normal na primeira chamada)")
except Exception as e:
    print(f"   ❌ ERRO: {e}")

# Teste 4: Verificar documentação
print("\n4. Verificando se a documentação está acessível...")
try:
    response = requests.get(f"{base_url}/docs")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   ✅ Documentação acessível em: " + f"{base_url}/docs")
except Exception as e:
    print(f"   ERRO: {e}")

print("\n" + "=" * 60)
print("TESTE COMPLETO")
print("=" * 60)