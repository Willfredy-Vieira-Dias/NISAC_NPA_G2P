from src.gemini_integracao_api_npa import app

# Export the app variable for Gunicorn
__all__ = ['app']

if __name__ == "__main__":
    import os
    import uvicorn
    
    # Use Render's PORT environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)