"""Patch vietocr for Python 3.14 + NumPy 2.x + no albumentations/imgaug."""
import sys, re, pathlib
sys.stdout.reconfigure(encoding="utf-8")

import vietocr
root = pathlib.Path(vietocr.__file__).parent

patches = {
    # 1. aug.py: stub ImgAugTransformV2 (albumentations/imgaug removed)
    root / "loader" / "aug.py": (
        None,  # replace entire file
        '''# Patched: albumentations/imgaug removed (Python 3.14 incompatible)
class ImgAugTransform:
    def __call__(self, img): return img

class ImgAugTransformV2:
    """No-op augmentor stub."""
    def __call__(self, img): return img
'''
    ),

    # 2. create_dataset.py: np.fromstring removed in NumPy 2.x → frombuffer
    root / "tool" / "create_dataset.py": (
        "np.fromstring(imageBin, dtype=np.uint8)",
        "np.frombuffer(imageBin, dtype=np.uint8)"
    ),

    # 3. dataloader.py: np.fromstring removed in NumPy 2.x → frombuffer
    root / "loader" / "dataloader.py": (
        "np.fromstring(dim_img, dtype=np.int32)",
        "np.frombuffer(dim_img, dtype=np.int32)"
    ),
}

for filepath, (old, new) in patches.items():
    if not filepath.exists():
        print(f"NOT FOUND: {filepath}")
        continue
    if old is None:
        # Replace entire file
        filepath.write_text(new, encoding="utf-8")
        print(f"Replaced : {filepath.name}")
    else:
        txt = filepath.read_text(encoding="utf-8")
        if old not in txt:
            print(f"Already patched or not found pattern: {filepath.name}")
        else:
            filepath.write_text(txt.replace(old, new), encoding="utf-8")
            print(f"Patched  : {filepath.name}")

print("Done.")
