import sys, os
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print('sys.path[0]=', sys.path[0])
try:
    from src.analise_clima import analisar_dados_climaticos
    print('SUCCESS: imported analisar_dados_climaticos')
except Exception as e:
    import traceback
    print('FAILED IMPORT:', repr(e))
    traceback.print_exc()
