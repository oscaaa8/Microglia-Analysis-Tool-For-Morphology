from __future__ import annotations

from pathlib import Path

import numpy as np
from data_io.load_czi import load_czi_image

###############################################################################
# Quick Config (Most-Changed)
###############################################################################
raw_cmap = "gray"  # "gray" or "plasma" for the left MIP panel.

###############################################################################
# Config
###############################################################################
file_path = "glia_data\\1012"  # Example - later this comes from your UI.
user_thresh = 0.96  # Initial threshold (znorm range [0,1]).
mask_color = (1.0, 0.35, 0.0, 0.6)  # Selected ROI overlay RGBA.
show_mask_default = True  # True shows the white mask; toggle with V.
raw_clim_percentiles = (1, 99)  # Contrast stretch for MIP display.
figure_size = (10, 5)  # Matplotlib figure size (inches).
figure_dpi = 100  # Matplotlib figure DPI.
update_debounce_ms = 150  # UI debounce for threshold/size changes.

# Output root (controls where skeletons + stats are saved)
output_root = Path("ROI_Save")
morph_skeletons_root = output_root / "Morph_Skeletons"
stats_root = output_root / "Processed Stats"

# Debug toggles:
# - enable_loading: UI opens without loading any data if False.
# - enable_znorm: disable znorm to isolate crashes.
# - enable_render: skip matplotlib image updates to isolate crashes.
enable_loading = True
enable_znorm = True
enable_render = True

use_cache = True
precache_czi = False
cache_root = output_root / "CZI_Cache"


def main() -> None:
    data_dir = Path(file_path)

    if use_cache and precache_czi:
        try:
            print("[cache] precache start")
            cache_root.mkdir(parents=True, exist_ok=True)
            czi_files = sorted(data_dir.glob("*.czi"))
            if not czi_files:
                print("[cache] no .czi files found to cache")
            for czi_path in czi_files:
                cache_path = cache_root / f"{czi_path.stem}.npy"
                if cache_path.exists():
                    print(f"[cache] hit {cache_path.name}")
                    continue
                print(f"[cache] loading {czi_path.name}")
                stack = load_czi_image(czi_path)
                print(f"[cache] saving {cache_path.name}")
                # Save as .npy for fast and stable UI loading.
                np.save(cache_path, stack)
            print("[cache] precache done")
        except Exception as exc:  # noqa: BLE001
            print(f"[cache] precache failed: {exc}")
            # If precache fails, keep UI open without loading.
            global enable_loading
            enable_loading = False

    # Import Qt only after any CZI I/O to avoid native library conflicts.
    from qtpy import QtWidgets

    qapp = QtWidgets.QApplication.instance()
    if qapp is None:
        qapp = QtWidgets.QApplication([])
    QtWidgets.QApplication.setQuitOnLastWindowClosed(True)
    qapp.aboutToQuit.connect(lambda: print("[qt] aboutToQuit"))

    # Import UI after QApplication is created to avoid Qt backend issues.
    from ui.qt_view import MicrogliaQtApp

    window = MicrogliaQtApp(
        data_dir=data_dir,
        morph_skeletons_root=morph_skeletons_root,
        stats_root=stats_root,
        user_thresh=user_thresh,
        thresh_min=0.900,
        thresh_max=1.000,
        min_object_area=5000,
        enable_loading=enable_loading,
        enable_znorm=enable_znorm,
        enable_render=enable_render,
        use_cache=use_cache,
        cache_root=cache_root,
        mask_color=mask_color,
        show_mask_default=show_mask_default,
        raw_clim_percentiles=raw_clim_percentiles,
        raw_cmap=raw_cmap,
        figure_size=figure_size,
        figure_dpi=figure_dpi,
        update_debounce_ms=update_debounce_ms,
    )
    window.show()
    window.raise_()
    window.activateWindow()
    window.start()
    globals()["_MICROGLIA_QT_WINDOW"] = window
    qapp.exec_()


if __name__ == "__main__":
    main()
