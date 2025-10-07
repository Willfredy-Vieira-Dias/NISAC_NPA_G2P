# main.py
import uvicorn
from src.gemini_integracao_api_npa import app

# Esta verificação __name__ == "__main__" é opcional aqui, 
# mas é uma boa prática. O Gunicorn não a executará diretamente.
if __name__ == "__main__":
    # Use as configurações do Render ou valores padrão para a porta e host
    # O Render define a variável de ambiente PORT
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)