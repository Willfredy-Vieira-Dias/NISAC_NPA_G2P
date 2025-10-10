import asyncio
import sys
import os
from pathlib import Path
from types import ModuleType

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Create a minimal fake `fastapi` module if it's not installed in the test env
if 'fastapi' not in sys.modules:
    fake_fastapi = ModuleType('fastapi')
    # minimal placeholders
    class DummyApp:
        def __init__(self, *args, **kwargs):
            pass
        def add_middleware(self, *args, **kwargs):
            return None
        # provide common route decorators used at import time
        def _route_decorator(self, *dargs, **dkwargs):
            def _decor(fn):
                return fn
            return _decor
        def post(self, *args, **kwargs):
            return self._route_decorator(*args, **kwargs)
        def get(self, *args, **kwargs):
            return self._route_decorator(*args, **kwargs)
        def put(self, *args, **kwargs):
            return self._route_decorator(*args, **kwargs)
        def delete(self, *args, **kwargs):
            return self._route_decorator(*args, **kwargs)
    fake_fastapi.FastAPI = DummyApp
    fake_fastapi.HTTPException = Exception
    # middleware submodule
    middleware = ModuleType('fastapi.middleware')
    cors = ModuleType('fastapi.middleware.cors')
    def CORSMiddleware(*args, **kwargs):
        return None
    cors.CORSMiddleware = CORSMiddleware
    middleware.cors = cors
    fake_fastapi.middleware = middleware
    # responses
    responses = ModuleType('fastapi.responses')
    class JSONResponse:
        def __init__(self, content=None, media_type=None, headers=None):
            self.content = content
    responses.JSONResponse = JSONResponse
    fake_fastapi.responses = responses
    sys.modules['fastapi'] = fake_fastapi
    sys.modules['fastapi.middleware'] = middleware
    sys.modules['fastapi.middleware.cors'] = cors
    sys.modules['fastapi.responses'] = responses

# Ensure required env vars for Settings so module import won't fail
os.environ.setdefault('GEMINI_API_KEY', 'test_key')

import pytest
from src.gemini_integracao_api_npa import call_gemini_and_validate, RecommendationModel


class MockResponse:
    def __init__(self, text):
        self.text = text


class MockGemini:
    def __init__(self, responses):
        self._responses = responses
        self._i = 0

    async def generate_content_async(self, prompt, **kwargs):
        # return next response
        text = self._responses[self._i]
        self._i = min(self._i + 1, len(self._responses) - 1)
        await asyncio.sleep(0)
        return MockResponse(text)


@pytest.mark.asyncio
async def test_call_gemini_and_validate_success(monkeypatch, tmp_path):
    # Good JSON on first try
    good_json = '{"resumo_executivo": "Ok", "nivel_risco": "BAIXO", "recomendacao_principal": "Nada", "preparacoes_essenciais": [], "alternativas_sugeridas": {"datas_alternativas": [], "horarios_alternativos": "manhã", "tipo_local": "ambiente aberto"}, "itens_necessarios": [], "avisos_especiais": [], "dica_especial": ""}'
    mock = MockGemini([good_json])
    monkeypatch.setattr('src.gemini_integracao_api_npa.gemini_model', mock)

    result = await call_gemini_and_validate("dummy prompt", RecommendationModel)
    assert isinstance(result, dict)
    assert result['nivel_risco'] == 'BAIXO'


@pytest.mark.asyncio
async def test_call_gemini_and_validate_retry(monkeypatch):
    # Bad JSON first, then corrected
    bad = 'not a json'
    corrected = '{"resumo_executivo": "Ok", "nivel_risco": "MODERADO", "recomendacao_principal": "Cuidado", "preparacoes_essenciais": [], "alternativas_sugeridas": {"datas_alternativas": [], "horarios_alternativos": "tarde", "tipo_local": "ambiente fechado"}, "itens_necessarios": [], "avisos_especiais": [], "dica_especial": ""}'
    mock = MockGemini([bad, corrected])
    monkeypatch.setattr('src.gemini_integracao_api_npa.gemini_model', mock)

    result = await call_gemini_and_validate("dummy prompt", RecommendationModel)
    assert isinstance(result, dict)
    assert result['nivel_risco'] == 'MODERADO'
