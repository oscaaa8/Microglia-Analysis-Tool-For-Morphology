from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import traceback

import numpy as np
from qtpy import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage.measure import label

from data_io.load_czi import load_czi_image
from processing.algorithms import znorm, get_prop
from processing.skeletonization import save_skeleton_image


def _remove_small_objects_3d(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return mask.astype(bool)

    labeled = label(mask, connectivity=3)
    if labeled.max() == 0:
        return mask.astype(bool)

    sizes = np.bincount(labeled.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labeled]


@dataclass
class ProcessedRow:
    filename: str
    roi_number: int
    roi_id: int
    n_branch_points: float
    total_area: float
    mean_branch_length: float
    branch_depth: float
    cell_body_size: float

    def to_dict(self) -> dict[str, object]:
        return {
            "Filename": self.filename,
            "ROI_Number": self.roi_number,
            "Number of branch points": self.n_branch_points,
            "Total Area": self.total_area,
            "Mean Branch Length": self.mean_branch_length,
            "Branch Depth": self.branch_depth,
            "Cell Body Size": self.cell_body_size,
        }


class MicrogliaQtApp(QtWidgets.QMainWindow):
    def __init__(
        self,
        *,
        data_dir: Path,
        morph_skeletons_root: Path,
        stats_root: Path,
        user_thresh: float,
        thresh_min: float,
        thresh_max: float,
        min_object_area: int,
        enable_loading: bool = True,
        enable_render: bool = True,
        enable_znorm: bool = True,
        use_cache: bool = False,
        cache_root: Path | None = None,
        mask_color: tuple[float, float, float, float] = (1.0, 0.35, 0.0, 0.6),
        show_mask_default: bool = True,
        raw_clim_percentiles: tuple[float, float] = (1, 99),
        raw_cmap: str = "gray",
        figure_size: tuple[float, float] = (10, 5),
        figure_dpi: int = 100,
        update_debounce_ms: int = 150,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.morph_skeletons_root = Path(morph_skeletons_root)
        self.stats_root = Path(stats_root)
        self.user_thresh = float(user_thresh)
        self.thresh_min = float(thresh_min)
        self.thresh_max = float(thresh_max)
        self.thresh_min_ui = self.thresh_min
        self.thresh_max_ui = self.thresh_max
        self.min_object_area = int(min_object_area)
        self.enable_loading = bool(enable_loading)
        self.enable_render = bool(enable_render)
        self.enable_znorm = bool(enable_znorm)
        self.use_cache = bool(use_cache)
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.mask_color = tuple(float(x) for x in mask_color)
        self.show_mask = bool(show_mask_default)
        self.raw_clim_percentiles = raw_clim_percentiles
        self.raw_cmap = str(raw_cmap)
        self.figure_size = figure_size
        self.figure_dpi = int(figure_dpi)
        self.update_debounce_ms = int(update_debounce_ms)

        self.czi_files = sorted(self.data_dir.glob("*.czi"))
        if not self.czi_files:
            raise FileNotFoundError(f"No CZI files found in {self.data_dir}")

        self.current_index = 0
        self.roi_counter = 0
        self.processed_rows: list[ProcessedRow] = []
        self.selected_roi_id = 0
        self.last_click_xy: tuple[int, int] | None = None
        self.roi_ids: list[int] = []
        self.roi_index: int = -1

        self.stack: np.ndarray | None = None
        self.nz: np.ndarray | None = None
        self.current_mask: np.ndarray | None = None
        self.current_labels: np.ndarray | None = None
        self.label_mip: np.ndarray | None = None
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(self.update_debounce_ms)
        self._update_timer.timeout.connect(self._refresh_mask_view)

        self._build_ui()
        self._bind_shortcuts()

    # ------------------------------------------------------------------
    # App bootstrap
    # ------------------------------------------------------------------
    def run(self) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        self.show()
        self.start()
        app.exec_()

    def start(self) -> None:
        if self.enable_loading:
            QtCore.QTimer.singleShot(0, self._safe_load_current_file)
        else:
            self._set_status("Loading disabled (debug mode).")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setWindowTitle("Microglia Morph Analysis")
        self.resize(1400, 900)

        self._apply_style()

        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(10, 10, 10, 10)

        self.fig = Figure(figsize=self.figure_size, dpi=self.figure_dpi)
        self.canvas = FigureCanvas(self.fig)
        self.ax_raw = self.fig.add_subplot(1, 2, 1)
        self.ax_mask = self.fig.add_subplot(1, 2, 2)
        self.fig.tight_layout(pad=2.0)

        central_layout.addWidget(self.canvas)
        self.setCentralWidget(central)

        self._build_controls_dock()
        self._init_plot_artists()
        # ROI selection is driven by keyboard (A/D). Click selection disabled.

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #0f1216; }
            QLabel { color: #e6edf3; font-size: 12px; }
            QDockWidget { background: #11161c; border: 1px solid #1f2a35; }
            QDockWidget::title { padding: 8px; color: #e6edf3; background: #131a22; }
            QPushButton {
                background: #1f6feb;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover { background: #2b7ef3; }
            QPushButton:pressed { background: #1858c7; }
            QSlider::groove:horizontal { height: 6px; background: #1f2a35; border-radius: 3px; }
            QSlider::handle:horizontal {
                width: 14px; margin: -6px 0;
                background: #9cc3ff; border-radius: 7px;
            }
            QSpinBox, QDoubleSpinBox {
                background: #0f1216;
                color: #e6edf3;
                border: 1px solid #2a3644;
                border-radius: 4px;
                padding: 4px 6px;
            }
            QListWidget {
                background: #0f1216;
                color: #e6edf3;
                border: 1px solid #2a3644;
                border-radius: 6px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 6px 8px;
            }
            QListWidget::item:selected {
                background: #193a6a;
            }
            """
        )

    def _build_controls_dock(self) -> None:
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)

        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Microglia Morph Analysis")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        layout.addWidget(title)

        self.file_label = QtWidgets.QLabel("File: -")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

        self.status_label = QtWidgets.QLabel("Status: Ready")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        thresh_row = QtWidgets.QHBoxLayout()
        thresh_label = QtWidgets.QLabel("Threshold")
        self.thresh_spin = QtWidgets.QDoubleSpinBox()
        self.thresh_spin.setDecimals(3)
        self.thresh_spin.setRange(self.thresh_min_ui, self.thresh_max_ui)
        self.thresh_spin.setSingleStep(0.001)
        self.thresh_spin.setValue(self.user_thresh)
        self.thresh_spin.valueChanged.connect(self._on_thresh_change)
        thresh_row.addWidget(thresh_label)
        thresh_row.addWidget(self.thresh_spin)
        layout.addLayout(thresh_row)

        self.thresh_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.thresh_slider.setRange(0, 1000)
        self.thresh_slider.setValue(self._thresh_to_slider(self.user_thresh))
        self.thresh_slider.valueChanged.connect(self._on_thresh_slider)
        layout.addWidget(self.thresh_slider)

        size_row = QtWidgets.QHBoxLayout()
        size_label = QtWidgets.QLabel("Min Object")
        self.size_spin = QtWidgets.QSpinBox()
        self.size_spin.setRange(0, 10_000_000)
        self.size_spin.setValue(self.min_object_area)
        self.size_spin.valueChanged.connect(self._on_min_size_change)
        size_row.addWidget(size_label)
        size_row.addWidget(self.size_spin)
        layout.addLayout(size_row)

        self.process_btn = QtWidgets.QPushButton("Process ROI (P)")
        self.process_btn.clicked.connect(self._process_selected)
        self.next_btn = QtWidgets.QPushButton("Next File (Q)")
        self.next_btn.clicked.connect(self._next_file)
        self.save_btn = QtWidgets.QPushButton("Save CSV")
        self.save_btn.clicked.connect(self._write_csv)
        self.reset_btn = QtWidgets.QPushButton("Reset ROIs")
        self.reset_btn.clicked.connect(self._reset_current_rois)
        layout.addWidget(self.process_btn)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.reset_btn)

        self.session_label = QtWidgets.QLabel("Session: -")
        self.session_label.setWordWrap(True)
        layout.addWidget(self.session_label)

        roi_header = QtWidgets.QLabel("Analyzed ROIs")
        roi_header.setStyleSheet("font-weight: 600; margin-top: 4px;")
        layout.addWidget(roi_header)

        self.roi_list = QtWidgets.QListWidget()
        self.roi_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.roi_list.itemClicked.connect(self._on_roi_list_click)
        self.roi_list.setMinimumHeight(140)
        layout.addWidget(self.roi_list)

        hint = QtWidgets.QLabel("Click ROI on the right panel. W/S adjust threshold.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addStretch(1)
        dock.setWidget(root)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _init_plot_artists(self) -> None:
        self.ax_raw.set_title("MIP (Raw)")
        self.ax_raw.axis("off")
        self.ax_mask.set_title("Mask / ROI")
        self.ax_mask.axis("off")

        blank = np.zeros((10, 10))
        self.raw_im = self.ax_raw.imshow(blank, cmap=self.raw_cmap)
        self.mask_im = self.ax_mask.imshow(blank, cmap="gray")
        self.overlay_im = self.ax_mask.imshow(
            np.zeros((10, 10, 4), dtype=np.float32),
            interpolation="nearest",
        )

    # ------------------------------------------------------------------
    # Data + view updates
    # ------------------------------------------------------------------
    def _load_current_file(self) -> None:
        self.selected_roi_id = 0
        self.last_click_xy = None
        self.roi_counter = 0
        self.roi_list.clear()

        czi_path = self.czi_files[self.current_index]
        self.file_label.setText(f"File: {czi_path.name}")
        self._set_status("Loading stack...")

        if self.use_cache and self.cache_root is not None:
            cache_path = self.cache_root / f"{czi_path.stem}.npy"
            if cache_path.exists():
                self.stack = np.load(cache_path)
            else:
                self.stack = load_czi_image(czi_path)
        else:
            self.stack = load_czi_image(czi_path)
        if self.enable_znorm:
            self.nz = znorm(self.stack)
        else:
            self.nz = None

        self._update_mask()
        raw_mip = np.nanmax(self.stack, axis=0)

        if self.enable_render:
            self.raw_im.set_data(raw_mip)
            p_low, p_high = self.raw_clim_percentiles
            self.raw_im.set_clim(np.percentile(raw_mip, p_low), np.percentile(raw_mip, p_high))

            mask_mip = self._mask_mip()
            self.mask_im.set_data(mask_mip)
            self.mask_im.set_clim(0.0, 1.0)
            self.mask_im.set_visible(self.show_mask)
            self.overlay_im.set_data(self._overlay_rgba())

            self.ax_raw.set_title("MIP (Raw)")
            self.ax_mask.set_title(f"Mask (thresh={self.user_thresh:.3f})")
            self.canvas.draw_idle()

        self._refresh_session_panel()
        self._set_status("Ready")

    def _safe_load_current_file(self) -> None:
        try:
            self._load_current_file()
        except Exception as exc:  # noqa: BLE001
            print("[qt] load failed")
            print(traceback.format_exc())
            self._set_status(f"Load failed: {exc}")
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load file:\n{exc}",
            )

    def _update_mask(self) -> None:
        if self.nz is None:
            self.current_mask = None
            self.current_labels = None
            self.label_mip = None
            return
        raw_mask = self.nz > self.user_thresh
        current_mask = raw_mask
        if self.min_object_area > 0:
            filtered = _remove_small_objects_3d(raw_mask, self.min_object_area)
            # If filtering removes everything, fall back to raw for visibility.
            if np.any(filtered):
                current_mask = filtered
            else:
                self._set_status(
                    f"Mask empty after min size={self.min_object_area}; showing raw mask"
                )
        self.current_mask = current_mask
        self.current_labels = label(current_mask, connectivity=3)
        self.label_mip = np.nanmax(self.current_labels, axis=0)
        self._rebuild_roi_index()
        if not np.any(current_mask):
            self._set_status(f"Mask empty at thresh={self.user_thresh:.3f}")

    def _mask_mip(self) -> np.ndarray:
        if self.current_mask is None:
            return np.zeros((1, 1), dtype=np.float32)
        return np.nanmax(self.current_mask, axis=0).astype(np.float32)

    def _overlay_rgba(self) -> np.ndarray:
        if self.current_mask is None:
            return np.zeros((1, 1, 4), dtype=np.float32)

        h, w = self.current_mask.shape[1], self.current_mask.shape[2]
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        if self.selected_roi_id == 0 or self.label_mip is None:
            return overlay

        sel = self.label_mip == self.selected_roi_id
        overlay[sel, 0] = self.mask_color[0]
        overlay[sel, 1] = self.mask_color[1]
        overlay[sel, 2] = self.mask_color[2]
        overlay[sel, 3] = self.mask_color[3]
        return overlay

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax_mask:
            return
        if self.current_labels is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if (
            y < 0
            or x < 0
            or y >= self.current_labels.shape[1]
            or x >= self.current_labels.shape[2]
        ):
            return
        h = self.current_labels.shape[1]
        w = self.current_labels.shape[2]
        if y < 0 or x < 0 or y >= h or x >= w:
            return

        self.last_click_xy = (x, y)
        roi_id = int(np.max(self.current_labels[:, y, x]))

        # If empty, search a small neighborhood for the nearest labeled column.
        if roi_id == 0:
            found = False
            for dy in (-2, -1, 0, 1, 2):
                for dx in (-2, -1, 0, 1, 2):
                    yy = y + dy
                    xx = x + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        cand = int(np.max(self.current_labels[:, yy, xx]))
                        if cand != 0:
                            roi_id = cand
                            self.last_click_xy = (xx, yy)
                            found = True
                            break
                if found:
                    break

        self.selected_roi_id = roi_id
        self._sync_roi_index()
        self.overlay_im.set_data(self._overlay_rgba())
        self.canvas.draw_idle()
        self._refresh_session_panel()
        if self.selected_roi_id == 0:
            self._set_status("No ROI at clicked location.")
        else:
            self._set_status(f"Selected ROI id={self.selected_roi_id}")

    def _on_thresh_change(self, value: float) -> None:
        self.user_thresh = float(value)
        self.thresh_slider.blockSignals(True)
        self.thresh_slider.setValue(self._thresh_to_slider(self.user_thresh))
        self.thresh_slider.blockSignals(False)
        self._update_timer.start()

    def _on_thresh_slider(self, value: int) -> None:
        self.user_thresh = self._slider_to_thresh(value)
        self.thresh_spin.blockSignals(True)
        self.thresh_spin.setValue(self.user_thresh)
        self.thresh_spin.blockSignals(False)
        self._update_timer.start()

    def _on_min_size_change(self, value: int) -> None:
        self.min_object_area = int(value)
        self._update_timer.start()

    def _refresh_mask_view(self) -> None:
        if self.nz is None:
            return
        self._update_mask()
        # Try to keep selection by ROI id when threshold changes.
        if self.selected_roi_id != 0 and self.selected_roi_id in self.roi_ids:
            self._sync_roi_index()
        elif self.roi_ids:
            self.roi_index = min(self.roi_index, len(self.roi_ids) - 1)
            if self.roi_index < 0:
                self.roi_index = 0
            self._select_roi_by_index(self.roi_index)
        else:
            self.selected_roi_id = 0
        mask_mip = self._mask_mip()
        self.mask_im.set_data(mask_mip)
        self.mask_im.set_clim(0.0, 1.0)
        self.mask_im.set_visible(self.show_mask)
        self.overlay_im.set_data(self._overlay_rgba())
        self.ax_mask.set_title(f"Mask (thresh={self.user_thresh:.3f})")
        self.canvas.draw_idle()
        self._refresh_session_panel()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _process_selected(self) -> None:
        if self.selected_roi_id == 0 or self.current_labels is None:
            self._set_status("No ROI selected.")
            return

        selected_roi_id = int(self.selected_roi_id)
        selected_roi_mask = self.current_labels == self.selected_roi_id
        props = get_prop(selected_roi_mask)
        self.roi_counter += 1

        czi_path = self.czi_files[self.current_index]
        save_dir = self.morph_skeletons_root / czi_path.stem
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"ROI_{self.roi_counter}.tiff"
        save_skeleton_image(props.debug["bwcb"], props.debug["cell_body"], out_path)

        self.processed_rows.append(
            ProcessedRow(
                filename=czi_path.stem,
                roi_number=self.roi_counter,
                roi_id=selected_roi_id,
                n_branch_points=float(props.n_branch_points),
                total_area=float(props.total_area),
                mean_branch_length=float(props.mean_branch_length),
                branch_depth=float(props.branch_depth),
                cell_body_size=float(props.cell_body_size),
            )
        )
        self._append_roi_list_item(self.roi_counter, selected_roi_id, out_path.name)
        self._refresh_session_panel()
        self._set_status(f"Saved ROI {self.roi_counter} -> {out_path.name}")

    def _next_file(self) -> None:
        if self.current_index + 1 >= len(self.czi_files):
            finished_name = self.czi_files[self.current_index].name
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{finished_name} finished analyzing: {timestamp}")
            self._set_status("End of folder. Closing...")
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.quit()
            self.close()
            return
        self.current_index += 1
        self._load_current_file()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _write_csv(self) -> None:
        if not self.processed_rows:
            self._set_status("No rows to export.")
            return
        self.stats_root.mkdir(parents=True, exist_ok=True)
        out_name = f"ROI_Stats_{self.data_dir.name}.csv"
        out_path = self.stats_root / out_name
        fieldnames = [
            "Filename",
            "ROI_Number",
            "Number of branch points",
            "Total Area",
            "Mean Branch Length",
            "Branch Depth",
            "Cell Body Size",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([row.to_dict() for row in self.processed_rows])
        self._set_status(f"Saved CSV -> {out_path.name}")

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    def _set_status(self, message: str) -> None:
        self.status_label.setText(f"Status: {message}")

    def _refresh_session_panel(self) -> None:
        czi_path = self.czi_files[self.current_index]
        total_rois = len(self.roi_ids)
        analyzed = sum(1 for row in self.processed_rows if row.filename == czi_path.stem)
        selected = self.selected_roi_id if self.selected_roi_id != 0 else "-"
        file_pos = f"{self.current_index + 1}/{len(self.czi_files)}"
        self.session_label.setText(
            f"Session: {analyzed} analyzed / {total_rois} ROIs | Selected id={selected} | File {file_pos}"
        )

    def _append_roi_list_item(self, roi_number: int, roi_id: int, out_name: str) -> None:
        label = f"ROI {roi_number} (id {roi_id}) • {out_name}"
        item = QtWidgets.QListWidgetItem(label)
        item.setData(QtCore.Qt.UserRole, roi_id)
        self.roi_list.addItem(item)
        self.roi_list.scrollToBottom()

    def _on_roi_list_click(self, item: QtWidgets.QListWidgetItem) -> None:
        roi_id = item.data(QtCore.Qt.UserRole)
        if roi_id is None:
            return
        if self.current_labels is None:
            return
        if int(roi_id) not in self.roi_ids:
            self._set_status(f"ROI id={roi_id} not in current mask.")
            return
        self.selected_roi_id = int(roi_id)
        self._sync_roi_index()
        self.overlay_im.set_data(self._overlay_rgba())
        self.canvas.draw_idle()
        self._refresh_session_panel()
        self._set_status(f"Selected ROI id={self.selected_roi_id}")

    def _bind_shortcuts(self) -> None:
        QtWidgets.QShortcut(QtGui.QKeySequence("W"), self, activated=self._inc_thresh)
        QtWidgets.QShortcut(QtGui.QKeySequence("S"), self, activated=self._dec_thresh)
        QtWidgets.QShortcut(QtGui.QKeySequence("P"), self, activated=self._process_selected)
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self, activated=self._next_file)
        QtWidgets.QShortcut(QtGui.QKeySequence("A"), self, activated=self._prev_roi)
        QtWidgets.QShortcut(QtGui.QKeySequence("D"), self, activated=self._next_roi)
        QtWidgets.QShortcut(QtGui.QKeySequence("V"), self, activated=self._toggle_mask)

    def _inc_thresh(self) -> None:
        self.thresh_spin.setValue(min(self.user_thresh + 0.001, self.thresh_max))

    def _dec_thresh(self) -> None:
        self.thresh_spin.setValue(max(self.user_thresh - 0.001, self.thresh_min))

    def _rebuild_roi_index(self) -> None:
        if self.current_labels is None:
            self.roi_ids = []
            self.roi_index = -1
            return
        ids = np.unique(self.current_labels)
        ids = ids[ids != 0]
        self.roi_ids = [int(x) for x in ids]
        if not self.roi_ids:
            self.roi_index = -1

    def _sync_roi_index(self) -> None:
        if self.selected_roi_id in self.roi_ids:
            self.roi_index = self.roi_ids.index(self.selected_roi_id)
        else:
            self.roi_index = -1

    def _select_roi_by_index(self, index: int) -> None:
        if not self.roi_ids:
            self.selected_roi_id = 0
            self.roi_index = -1
            return
        self.roi_index = max(0, min(index, len(self.roi_ids) - 1))
        self.selected_roi_id = self.roi_ids[self.roi_index]
        self.overlay_im.set_data(self._overlay_rgba())
        self.canvas.draw_idle()
        self._refresh_session_panel()
        self._set_status(f"Selected ROI id={self.selected_roi_id}")

    def _prev_roi(self) -> None:
        if not self.roi_ids:
            self._set_status("No ROIs available.")
            return
        if self.roi_index < 0:
            self.roi_index = 0
        else:
            self.roi_index = (self.roi_index - 1) % len(self.roi_ids)
        self._select_roi_by_index(self.roi_index)

    def _next_roi(self) -> None:
        if not self.roi_ids:
            self._set_status("No ROIs available.")
            return
        if self.roi_index < 0:
            self.roi_index = 0
        else:
            self.roi_index = (self.roi_index + 1) % len(self.roi_ids)
        self._select_roi_by_index(self.roi_index)

    def _thresh_to_slider(self, value: float) -> int:
        if self.thresh_max_ui == self.thresh_min_ui:
            return 0
        t = (value - self.thresh_min_ui) / (self.thresh_max_ui - self.thresh_min_ui)
        return int(round(t * 1000))

    def _slider_to_thresh(self, value: int) -> float:
        t = float(value) / 1000.0
        return self.thresh_min_ui + t * (self.thresh_max_ui - self.thresh_min_ui)

    def _toggle_mask(self) -> None:
        self.show_mask = not self.show_mask
        self.mask_im.set_visible(self.show_mask)
        self.canvas.draw_idle()
        state = "on" if self.show_mask else "off"
        self._set_status(f"Mask visibility {state}")

    def _reset_current_rois(self) -> None:
        czi_path = self.czi_files[self.current_index]
        save_dir = self.morph_skeletons_root / czi_path.stem
        if save_dir.exists():
            for path in save_dir.glob("*"):
                try:
                    path.unlink()
                except OSError:
                    pass
            try:
                save_dir.rmdir()
            except OSError:
                pass

        self.processed_rows = [
            row for row in self.processed_rows if row.filename != czi_path.stem
        ]
        self.roi_counter = 0
        self.roi_list.clear()
        self._refresh_session_panel()
        self._set_status("Reset current file ROI list.")

    # ------------------------------------------------------------------
    # Qt override
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802
        if self.processed_rows:
            self._write_csv()
        event.accept()
