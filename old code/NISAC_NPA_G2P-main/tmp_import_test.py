import importlib, sys, os
sys.path.insert(0, os.getcwd())
print('CWD', os.getcwd())
try:
    m = importlib.import_module('src.gemini_integracao_api_npa')
    print('Imported gemini module')
    from src.analise_clima import analisar_dados_climaticos
    print('Imported analisar_dados_climaticos from src.analise_clima')
except Exception as e:
    print('Import error:', repr(e))
