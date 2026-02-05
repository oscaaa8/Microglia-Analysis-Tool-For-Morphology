from pathlib import Path
from aicsimageio import AICSImage
import numpy as np


def load_czi_image(path_or_folder: Path) -> np.ndarray:
    """
    Load a CZI image and return the image stack.

    Parameters:
    - path_or_folder: Path to a CZI file or a folder containing CZI files.

    Returns:
    - stack: 3D numpy array representing the image stack (Z, Y, X).
    """
    path_or_folder = Path(path_or_folder)

    if path_or_folder.is_dir():
        czi_files = sorted(path_or_folder.glob("*.czi"))
        if not czi_files:
            raise FileNotFoundError(f"No CZI files found in {path_or_folder}")
        path = czi_files[0]
    else:
        path = path_or_folder

    if path.suffix.lower() != ".czi":
        raise ValueError(f"Expected a .czi file, got: {path}")

    print(f"Loading file: {path}")

    img = AICSImage(path)
    data = img.data
    stack = data[0, 0]  # shape: (Z, Y, X)
    print("Loaded stack shape:", stack.shape)

    return stack

''' 
Your stack is a 3D volume:

stack[z, y, x] = intensity value at that voxel

Z is slices (you saw ~30)

Y/X are the 2D image plane (~1024x1024 pixels)

'''
