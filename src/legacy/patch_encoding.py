import sys, pathlib
sys.stdout.reconfigure(encoding="utf-8")
import vietocr

f = pathlib.Path(vietocr.__file__).parent / "tool" / "create_dataset.py"
txt = f.read_text(encoding="utf-8")
patched = txt.replace("open(annotation_path, 'r')", "open(annotation_path, 'r', encoding='utf-8')")
f.write_text(patched, encoding="utf-8")
print("Patched:", f)
