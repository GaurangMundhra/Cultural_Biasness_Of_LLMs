import importlib, sys
pkgs = [
    ("python","sys.version"),
    ("torch","torch"),
    ("transformers","transformers"),
    ("sentence-transformers","sentence_transformers"),
    ("scipy","scipy"),
    ("streamlit","streamlit"),
    ("sklearn","sklearn"),
    ("plotly","plotly"),
]
print("Python:", sys.version.splitlines()[0])
ok = True
for name, mod in pkgs[1:]:
    try:
        importlib.import_module(mod)
        print(f"{name}: OK")
    except Exception as e:
        print(f"{name}: FAIL -> {e.__class__.__name__}: {e}")
        ok = False
if not ok:
    sys.exit(1)
print("All imports checked.")
