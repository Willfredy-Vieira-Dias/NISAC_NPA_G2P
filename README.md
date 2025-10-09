# NISAC_NPA_G2P

Small API that uses Gemini + NASA POWER to analyze historical weather probabilities for an event date. This README documents a recent import fix and how to run the project locally or on Render.

## Problem observed
When starting on certain platforms (Render, Gunicorn, or when working directory differs), Python raised ImportError for:

    from src.analise_clima import analisar_dados_climaticos

The root cause is usually that the process' working directory or PYTHONPATH does not include the project root (the directory that contains the `src` package). If Python doesn't find the `src` package in sys.path, the import fails and the code fell back to a mock implementation.

## What was changed (quick summary)
- `src/gemini_integracao_api_npa.py` now:
  - Prints startup diagnostics: current working directory and first entries of `sys.path`.
  - Tries to import the analysis module using a relative import (`from .analise_clima import ...`) first.
  - Ensures the project root (parent of `src`) is inserted into `sys.path` before import attempts.
  - Falls back to `from src.analise_clima import ...` if the relative import fails.
  - Logs full exception details if the import still fails and keeps a mock fallback to allow the server to start.
  - Guards optional heavy external libs (e.g., `google.generativeai`) so a plain import won't fail when those aren't installed.

These changes make the module import robust for the main execution patterns encountered on hosting platforms.

## How to run locally (recommended)
1. Create and activate a virtual environment and install dependencies (or use conda environment provided).

With venv + pip (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then start the app locally from the project root (important — start from the repo root so `src` is discoverable):

```powershell
# option A: use the main runner
python main.py

# option B: run uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try the API.

## Recommended Render start command
In Render (or similar platforms) the important bit is ensuring the process sees the project root. Two safe options:

1) Explicit PYTHONPATH (recommended if you use gunicorn):

```
PYTHONPATH=. gunicorn -k uvicorn.workers.UvicornWorker main:app
```

2) Start via `python main.py` (this will call `uvicorn.run` in `main.py`) and Render will use the repo root as the working directory by default (confirm in Render settings):

```
python main.py
```

If you still see the message `Módulo 'analise_clima' não encontrado. Usando dados mock.` in Render logs, check the startup diagnostics printed by the app. They look like:

```
[startup] cwd=C:\path\to\app
[startup] sys.path (top 10)=['C:\path\to\app', '...']
```

Confirm that the path shown is the repository root containing `src`.

## If the issue persists
- Copy the startup `cwd` and `sys.path` values from the Render logs and share them here so I can advise further.
- Ensure `src` is present at that `cwd` (Render may be using a different build directory).
- Confirm that the start command in Render uses the repository root as working directory.
