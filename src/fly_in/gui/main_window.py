"""Desktop map library, simulation playback, canvas, and inspection window.

The window delegates route planning to a worker and animates completed turns on
the Qt event thread.
"""

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction, QColor, QCloseEvent
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QProgressBar, QPushButton, QSizePolicy, QSlider, QSplitter, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from fly_in.solver_factory import DEFAULT_SOLVER, SolverFactory
from fly_in.gui.animator import TurnAnimationController
from fly_in.gui.catalog import MapCatalog, MapFile
from fly_in.gui.inspector import InspectorDockWidget
from fly_in.gui.map_view import MapGraphicsView
from fly_in.gui.planning import PlanningWorker
from fly_in.gui.theme import STYLESHEET
from fly_in.models import MapConfig
from fly_in.parser import Parser
from fly_in.provenance import PlanningProvenance
from fly_in.runs import RunSession, RunStatus
from fly_in.simulation import Simulation


class MainWindow(QMainWindow):
    """Desktop window for map selection, planning, playback, and inspection.

    Attributes:
        map_root (Path): Directory scanned for map files.
        output_dir (Path): Directory for generated run output.
        records_dir (Path): Directory for solver records.
        solver_name (str): Solver selected for current and future runs.
        solver_parameters (dict[str, float | int]): Numeric solver parameters.
        simulation (Simulation | None): Mutable state for the current run.
        session (RunSession | None): Run metadata and artifact writer.
        animator (TurnAnimationController): Controller for canvas transitions.
        canvas (MapGraphicsView): Map topology and drone-marker view.
        inspector (InspectorDockWidget): Dock for pinned topology details.
    """

    def __init__(
        self, map_root: Path | None = None,
        initial_map: Path | None = None, solver_name: str = DEFAULT_SOLVER,
        output_dir: Path | None = None, records_dir: Path | None = None,
        animate: bool = True,
        solver_parameters: dict[str, float | int] | None = None,
    ) -> None:
        """Create the desktop shell and optionally load a map paused.

        Args:
            map_root (Path | None): Map library directory; defaults to
                ``cwd/maps``.
            initial_map (Path | None): Optional map file to load after scanning
                the library.
            solver_name (str): Solver registry name used to plan each run.
            output_dir (Path | None): Run artifact directory; defaults to
                ``cwd/out``.
            records_dir (Path | None): Solver record directory; defaults to
                ``cwd/records``.
            animate (bool): Whether to interpolate movement between turns.
            solver_parameters (dict[str, float | int] | None): Optional numeric
                solver settings.
        """
        super().__init__()
        self.catalog = MapCatalog()
        self.map_root = (map_root or Path.cwd() / "maps").resolve()
        self.output_dir = (output_dir or Path.cwd() / "out").resolve()
        self.records_dir = (records_dir or Path.cwd() / "records").resolve()
        self.solver_name = solver_name
        self.solver_parameters = dict(solver_parameters or {})
        self.animate = animate
        self.map_entries: dict[str, MapFile] = {}
        self.selected_entry: MapFile | None = None
        self.loaded_config: MapConfig | None = None
        self.loaded_map_path: Path | None = None
        self._source_bytes = b""
        self.simulation: Simulation | None = None
        self._display_simulation: Simulation | None = None
        self.session: RunSession | None = None
        self._generation = 0
        self._workers: list[PlanningWorker] = []
        self._ready = False
        self._playing = False
        self._stopped = False
        self._closed = False
        self.animator = TurnAnimationController(self)
        self.animator.finished.connect(self._turn_finished)
        self.setWindowTitle("Fly-In Drones")
        self.resize(1440, 860)
        self.setMinimumSize(920, 620)
        self._build_ui()
        self.inspector = InspectorDockWidget(self)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.inspector
        )
        self.menuBar().addMenu("View").addAction(
            self.inspector.toggleViewAction()
        )
        for shortcut, callback in (
            ("Space", self._toggle_playback),
            ("Right", self._step_simulation),
            ("R", self._reset_simulation), ("F", self.map_view_fit),
        ):
            action = QAction(self)
            action.setShortcut(shortcut)
            action.triggered.connect(callback)
            self.addAction(action)
        self.setStyleSheet(STYLESHEET)
        self.reload_maps()
        if initial_map is not None:
            path = initial_map.expanduser().resolve()
            try:
                config = Parser().parse(path)
            except (ValueError, OSError) as error:
                self.statusBar().showMessage(str(error))
            else:
                self.selected_entry = MapFile(path, path.name, config)
                self.load_selected_map()

    def _build_ui(self) -> None:
        """Compose the map library, workspace, menus, and shortcuts."""
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        header = QWidget()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 12, 20, 12)
        brand = QLabel("FLY-IN  /  DRONE ROUTING")
        brand.setObjectName("brand")
        header_layout.addWidget(brand)
        self.root_label = QLabel()
        self.root_label.setTextFormat(Qt.TextFormat.PlainText)
        self.root_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.root_label.setObjectName("rootLabel")
        header_layout.addWidget(self.root_label, 1)
        self.folder_button = QPushButton("Map folder")
        self.folder_button.clicked.connect(self.choose_map_folder)
        header_layout.addWidget(self.folder_button)
        root_layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_workspace())
        splitter.setSizes([300, 920])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Choose a map to inspect its network.")
        reload_action = QAction("Reload maps", self)
        reload_action.setShortcut("Ctrl+R")
        reload_action.triggered.connect(self.reload_maps)
        self.addAction(reload_action)

    def _build_sidebar(self) -> QWidget:
        """Build the map picker and simulation controls.

        Returns:
            QWidget: Sidebar widget containing the map library and playback
                controls.
        """
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(260)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 20, 18, 14)
        layout.setSpacing(12)

        title_row = QVBoxLayout()
        title = QLabel("MAP LIBRARY")
        title.setObjectName("sectionTitle")
        title_row.addWidget(title)
        self.map_count = QLabel("0 maps")
        self.map_count.setObjectName("quietText")
        title_row.addWidget(self.map_count)
        layout.addLayout(title_row)

        self.map_tree = QTreeWidget()
        self.map_tree.setHeaderHidden(True)
        self.map_tree.setIndentation(15)
        self.map_tree.itemSelectionChanged.connect(self._selection_changed)
        self.map_tree.itemDoubleClicked.connect(
            lambda *_: self.load_selected_map()
        )
        layout.addWidget(self.map_tree, 1)

        action_row = QHBoxLayout()
        self.load_button = QPushButton("Load map")
        self.load_button.setObjectName("primaryButton")
        self.load_button.setEnabled(False)
        self.load_button.clicked.connect(self.load_selected_map)
        action_row.addWidget(self.load_button, 1)
        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload_maps)
        action_row.addWidget(self.reload_button)
        layout.addLayout(action_row)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setObjectName("separator")
        layout.addWidget(separator)

        sim_title = QLabel("SIMULATION")
        sim_title.setObjectName("sectionTitle")
        layout.addWidget(sim_title)
        stats = QVBoxLayout()
        stats.setSpacing(7)
        self.simulation_state = QLabel("No simulation running")
        self.simulation_state.setObjectName("simulationState")
        stats.addWidget(self.simulation_state)
        self.turn_label = QLabel("Turn      --")
        self.drone_label = QLabel("Drones   --")
        self.delivered_label = QLabel("Delivered --")
        self.in_flight_label = QLabel("In flight  --")
        self.waiting_label = QLabel("At hubs    --")
        for label in (
            self.turn_label,
            self.drone_label,
            self.delivered_label,
            self.in_flight_label,
            self.waiting_label,
        ):
            label.setObjectName("statLabel")
            stats.addWidget(label)
        layout.addLayout(stats)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        controls = QVBoxLayout()
        self.play_button = QPushButton("▶ Start Simulation")
        self.play_button.setObjectName("primaryButton")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._toggle_playback)
        controls.addWidget(self.play_button)
        self.step_button = QPushButton("Step")
        self.step_button.setEnabled(False)
        self.step_button.clicked.connect(self._step_simulation)
        controls.addWidget(self.step_button)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self._reset_simulation)
        controls.addWidget(self.reset_button)
        layout.addLayout(controls)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setInvertedAppearance(True)
        self.speed_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #3b4261;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal { background: #7dcfff; }
            QSlider::add-page:horizontal { background: #3b4261; }
            QSlider::handle:horizontal {
                background: #f8f8f2;
                border: 1px solid #7dcfff;
                width: 12px;
                margin: -6px 0;
                border-radius: 4px;
            }
        """)
        self.speed_slider.setRange(100, 2000)
        self.speed_slider.setSingleStep(100)
        self.speed_slider.setValue(500)
        self.speed_slider.setEnabled(False)
        self.speed_slider.setToolTip("Simulation speed")
        self.speed_slider.valueChanged.connect(self._update_play_speed)
        layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("500 ms / turn (2.0x)")
        layout.addWidget(self.speed_label)
        return sidebar

    def _build_workspace(self) -> QWidget:
        """Build the canvas, topology summary, and legend.

        Returns:
            QWidget: Workspace widget containing the map canvas and footer.
        """
        workspace = QWidget()
        workspace.setObjectName("workspace")
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(12)

        toolbar = QHBoxLayout()
        self.map_title = QLabel("Map preview")
        self.map_title.setTextFormat(Qt.TextFormat.PlainText)
        self.map_title.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.map_title.setObjectName("mapTitle")
        toolbar.addWidget(self.map_title, 1)
        self.map_summary = QLabel("Select a file from the library")
        self.map_summary.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.map_summary.setObjectName("quietText")
        toolbar.addWidget(self.map_summary)
        self.fit_button = QPushButton("Fit view")
        self.fit_button.clicked.connect(self.map_view_fit)
        toolbar.addWidget(self.fit_button)
        self.inspector_button = QPushButton("Inspector")
        self.inspector_button.setToolTip("Show the inspection pane")
        self.inspector_button.clicked.connect(self.show_inspector)
        toolbar.addWidget(self.inspector_button)
        layout.addLayout(toolbar)

        self.canvas = MapGraphicsView()
        self.canvas.setObjectName("mapCanvas")
        layout.addWidget(self.canvas, 1)

        footer = QHBoxLayout()
        self.selection_detail = QLabel(
            "Select a map to see its validation result."
        )
        self.selection_detail.setObjectName("detailText")
        self.selection_detail.setTextFormat(Qt.TextFormat.PlainText)
        self.selection_detail.setWordWrap(True)
        footer.addWidget(self.selection_detail, 1)
        self.canvas.object_selected.connect(self._object_selected)
        self.legend = QLabel(
            '<span style="color:#73daca">S</span> Start  '
            '<span style="color:#e0af68">G</span> Goal  '
            '<span style="color:#bb9af7">&lt;&gt;</span> Priority  '
            '<span style="color:#ff9e64">||</span> Restricted  '
            '<span style="color:#9aa5ce">X</span> Blocked'
        )
        self.legend.setObjectName("legend")
        footer.addWidget(self.legend, 0, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(footer)
        return workspace

    def reload_maps(self) -> None:
        """Cancel the active run and rescan the configured map directory."""
        self._discard_session()
        self.root_label.setText(str(self.map_root))
        self.root_label.setToolTip(str(self.map_root))
        self.map_tree.clear()
        self.map_entries.clear()
        try:
            entries = self.catalog.scan(self.map_root)
        except (OSError, NotADirectoryError) as error:
            self.selection_detail.setText(str(error))
            self.statusBar().showMessage(
                "Could not read the selected map folder."
            )
            self.load_button.setEnabled(False)
            return

        tree_nodes: dict[str, QTreeWidgetItem] = {}
        valid_count = 0
        for entry in entries:
            parts = Path(entry.relative_path).parts
            parent = self.map_tree.invisibleRootItem()
            parent_key = ""
            for folder in parts[:-1]:
                parent_key = f"{parent_key}/{folder}"
                if parent_key not in tree_nodes:
                    folder_item = QTreeWidgetItem([folder])
                    folder_item.setFlags(
                        folder_item.flags() & ~Qt.ItemFlag.ItemIsSelectable
                    )
                    parent.addChild(folder_item)
                    tree_nodes[parent_key] = folder_item
                parent = tree_nodes[parent_key]

            file_item = QTreeWidgetItem([parts[-1]])
            file_item.setData(0, Qt.ItemDataRole.UserRole, str(entry.path))
            file_item.setToolTip(0, entry.error or entry.relative_path)
            if entry.is_valid:
                file_item.setForeground(0, QColor("#9ece6a"))
                valid_count += 1
            else:
                file_item.setForeground(0, QColor("#f7768e"))
            parent.addChild(file_item)
            self.map_entries[str(entry.path)] = entry

        self.map_tree.expandAll()
        self.map_count.setText(f"{valid_count}/{len(entries)} ready")
        self.selected_entry = None
        self.load_button.setEnabled(False)
        if entries:
            self.statusBar().showMessage(
                f"Found {valid_count} valid map(s) and "
                f"{len(entries) - valid_count} with errors."
            )
        else:
            self.selection_detail.setText(
                "No .txt files found in this folder."
            )
            self.statusBar().showMessage("No map files found.")

    def choose_map_folder(self) -> None:
        """Prompt for a library directory and rescan it when selected."""
        folder = QFileDialog.getExistingDirectory(
            self, "Choose map folder", str(self.map_root)
        )
        if folder:
            self.map_root = Path(folder).resolve()
            self.reload_maps()

    def _selection_changed(self) -> None:
        """Update the selected entry and display its validation summary."""
        items = self.map_tree.selectedItems()
        if not items:
            self.selected_entry = None
            self.load_button.setEnabled(False)
            return
        path = items[0].data(0, Qt.ItemDataRole.UserRole)
        self.selected_entry = self.map_entries.get(path)
        if self.selected_entry is None:
            self.load_button.setEnabled(False)
            return
        self.load_button.setEnabled(self.selected_entry.is_valid)
        if self.selected_entry.error:
            self.selection_detail.setText(self.selected_entry.error)
            self.selection_detail.setStyleSheet("color: #f7768e")
            self.statusBar().showMessage("This map has a parsing error.")
        else:
            config = self.selected_entry.config
            assert config is not None
            self.selection_detail.setStyleSheet("")
            self.selection_detail.setText(
                f"{config.nb_drones} drones  ·  "
                f"{len(config.zones) + 2} zones  ·  "
                f"{len(config.connections)} links"
            )

    def load_selected_map(self) -> None:
        """Load the selected valid map at Turn 0 without starting playback."""
        entry = self.selected_entry
        if entry is None or entry.config is None:
            return
        try:
            source = entry.path.read_bytes()
            config = Parser().parse_text(source.decode("utf-8"))
        except (ValueError, OSError) as error:
            self.statusBar().showMessage(str(error))
            return
        self._load_config(config, entry.path, source)

    def _load_config(
        self, config: MapConfig, path: Path, source: bytes
    ) -> None:
        """Replace the active run and prepare isolated worker state.

        Args:
            config (MapConfig): Parsed map topology for the new run.
            path (Path): Source map path used for display and run metadata.
            source (bytes): Original UTF-8 map bytes retained with the run.
        """
        try:
            simulation = Simulation.from_config(
                config, SolverFactory(
                    path.stem, self.records_dir,
                ).create(self.solver_name, self.solver_parameters)
            )
        except ValueError as error:
            self.statusBar().showMessage(str(error))
            return
        self._discard_session()
        self.loaded_config = config
        self.loaded_map_path = path
        self._source_bytes = source
        self.simulation = simulation
        self.session = RunSession(
            path, self.solver_name, map_root=self.map_root,
            output_root=self.output_dir, records_root=self.records_dir,
            solver_parameters=self.solver_parameters,
            source_bytes=source,
        )
        self.canvas.show_map(config)
        self.canvas.show_drones(self.simulation.drones)
        self.canvas.fit_map()
        self.inspector.set_config(config)
        self._capture_display_state()
        self.map_title.setText(path.stem.replace("_", " ").title())
        self.map_summary.setText(
            f"{config.nb_drones} drones / {len(config.connections)} links"
        )
        self.selection_detail.setText(
            "Click a hub or connection to pin its live details."
        )
        self.simulation_state.setText("Preparing routes · Turn 0")
        self.statusBar().showMessage("Preparing routes in the background…")
        self.reset_button.setEnabled(True)
        self.speed_slider.setEnabled(True)
        self._refresh_simulation_stats()
        worker = PlanningWorker(
            self._generation, config, self.solver_name,
            self.solver_parameters, path.stem, self.records_dir,
        )
        self._workers.append(worker)
        worker.prepared.connect(self._prepared)
        worker.failed.connect(self._planning_failed)
        worker.finished.connect(self._collect_workers)
        worker.start()

    @Slot()
    def _collect_workers(self) -> None:
        """Delete finished worker objects later and retain active ones."""
        retained = []
        for worker in self._workers:
            if worker.isFinished():
                worker.deleteLater()
            else:
                retained.append(worker)
        self._workers = retained

    @Slot(int, object, float)
    def _prepared(
        self, generation: int, result: object, elapsed: float
    ) -> None:
        """Adopt a prepared simulation only if its generation is current.

        Args:
            generation (int): Worker generation associated with this result.
            result (object): Prepared simulation object emitted by the worker.
            elapsed (float): Planning duration in seconds.
        """
        if generation != self._generation or self._closed:
            return
        if not isinstance(result, Simulation):
            return
        planning = PlanningProvenance.capture(result.solver)
        if self.session is not None:
            self.session.planning_duration_seconds = elapsed
            self.session.planning_provenance = planning
        if self.solver_name == 'optimal' and not (
            planning.optimal_makespan and planning.status == 'optimal'
            and planning.proof_reason
        ):
            self._ready = False
            self._stop_with_error('minimum-turn planning did not finish proof')
            return
        self.simulation = result
        self._ready = True
        self.play_button.setEnabled(True)
        self.step_button.setEnabled(True)
        self.play_button.setText("▶ Start Simulation")
        self.simulation_state.setText("Ready to launch")
        self.statusBar().showMessage(
            f"Ready · {self.solver_name} · planning {elapsed:.3f}s"
        )
        self._capture_display_state()

    @Slot(int, str)
    def _planning_failed(self, generation: int, message: str) -> None:
        """Report a planning error for the current worker generation.

        Args:
            generation (int): Worker generation associated with the error.
            message (str): Error text to show in the status bar and run record.
        """
        if generation == self._generation and not self._closed:
            for worker in self._workers:
                if (worker.generation == generation
                        and self.session is not None):
                    self.session.planning_provenance = (
                        worker.planning_provenance
                    )
                    self.session.planning_duration_seconds = (
                        worker.planning_duration_seconds
                    )
                    break
            self._stop_with_error(message)

    def _discard_session(self) -> None:
        """Invalidate old callbacks, cancel work, and clear the current run."""
        self._generation += 1
        self._playing = False
        self._ready = False
        self._stopped = False
        self.animator.cancel()
        for worker in self._workers:
            worker.cancel()
        if self.session is not None and self.simulation is not None:
            self._save_run("cancelled")
        self.session = None
        self.simulation = None
        self._display_simulation = None
        self.loaded_config = None
        self.loaded_map_path = None
        self.canvas.clear_map()
        self.inspector.set_config(None)
        self.inspector.refresh(None)
        self.play_button.setText("▶ Start Simulation")
        self._style_play_button(False)
        for button in (self.play_button, self.step_button, self.reset_button):
            button.setEnabled(False)
        self.speed_slider.setEnabled(False)
        self.progress.setValue(0)
        self.simulation_state.setText("Choose a map")
        for label in (
            self.turn_label, self.drone_label, self.delivered_label,
            self.in_flight_label, self.waiting_label,
        ):
            label.setText("—")

    def _toggle_playback(self) -> None:
        """Start, resume, or pause playback when planning is ready."""
        if not self._ready or self._stopped:
            return
        if self._playing or (
            self.animator.is_running and not self.animator.is_paused
        ):
            self._playing = False
            self.animator.pause()
            self.play_button.setText("▶ Resume")
            self._style_play_button(False)
            self.simulation_state.setText("Paused")
            self.step_button.setEnabled(True)
            return
        self._playing = True
        self.play_button.setText("⏸ Pause")
        self._style_play_button(True)
        self.simulation_state.setText("Running")
        self.step_button.setEnabled(False)
        if self.animator.is_paused:
            self.animator.resume()
        elif not self.animator.is_running:
            self._advance_turn()

    def _update_play_speed(self, value: int | None = None) -> None:
        """Update the display for the chosen per-turn duration.

        Args:
            value (int | None): Duration in milliseconds from the slider
                signal, or None to read the slider's current value.
        """
        duration = self.speed_slider.value() if value is None else value
        self.speed_label.setText(
            f"{duration} ms / turn ({1000 / duration:.1f}x)"
        )

    def _style_play_button(self, playing: bool) -> None:
        """Set the play button's dynamic playing style.

        Args:
            playing (bool): Whether the button should use its active
                appearance.
        """
        self.play_button.setProperty("playing", playing)
        self.play_button.style().unpolish(self.play_button)
        self.play_button.style().polish(self.play_button)

    def _step_simulation(self) -> None:
        """Advance one turn when ready and not already playing."""
        if not self._ready or self._stopped or self._playing:
            return
        if self.animator.is_paused:
            self.step_button.setEnabled(False)
            self.play_button.setText("⏸ Pause")
            self._style_play_button(True)
            self.animator.resume()
            return
        if self.animator.is_running:
            return
        self._advance_turn()

    def _advance_turn(self) -> None:
        """Commit one legal turn and animate toward its resulting positions."""
        simulation = self.simulation
        if simulation is None or self._stopped or not self._ready:
            return
        self.step_button.setEnabled(False)
        self.play_button.setText("⏸ Pause")
        self._style_play_button(True)
        try:
            moves = simulation.step()
            pending = getattr(simulation.solver, "has_pending_actions", False)
            if (not moves and not pending
                    and not any(d.in_transit for d in simulation.drones)):
                raise RuntimeError("simulation is deadlocked")
        except (RuntimeError, ValueError) as error:
            self._stop_with_error(str(error))
            return
        targets = self.canvas.drone_positions(
            simulation.drones, include_delivered=True
        )
        self.statusBar().showMessage(
            f"Turn {simulation.turn}: {' '.join(moves) or 'waiting'}"
        )
        self.animator.start(
            self.canvas.drone_items, targets, self.speed_slider.value(),
            animate=self.animate,
        )

    @Slot()
    def _turn_finished(self) -> None:
        """Refresh the displayed snapshot and queue autoplay after a turn."""
        simulation = self.simulation
        if simulation is None or self._closed:
            return
        self.canvas.show_drones(simulation.drones)
        self.canvas.refresh(simulation)
        self._capture_display_state()
        self._refresh_simulation_stats()
        if not self._playing:
            self._style_play_button(False)
        if all(drone.delivered for drone in simulation.drones):
            self._playing = False
            self._style_play_button(False)
            self._stopped = True
            self.play_button.setText("✓ Completed")
            self.play_button.setEnabled(False)
            self.step_button.setEnabled(False)
            self.simulation_state.setText(
                f"Completed · {simulation.turn} turns"
            )
            self._save_run("completed")
        elif self._playing:
            generation = self._generation
            QTimer.singleShot(0, lambda: self._next_turn(generation))
        else:
            self.step_button.setEnabled(True)
            self.play_button.setText("▶ Resume")
            self.simulation_state.setText("Paused")

    def _next_turn(self, generation: int) -> None:
        """Advance playback only when the scheduled generation is current.

        Args:
            generation (int): Session generation captured when queuing the
                callback.
        """
        if generation == self._generation and self._playing:
            self._advance_turn()

    def _stop_with_error(self, message: str) -> None:
        """Stop a failed run, save partial history, and show the error.

        Args:
            message (str): Failure detail shown in the status bar and run
                record.
        """
        self._playing = False
        self._stopped = True
        self.animator.cancel()
        self._style_play_button(False)
        self.play_button.setEnabled(False)
        self.step_button.setEnabled(False)
        self.simulation_state.setText("Simulation stopped")
        self._save_run("failed", message)
        self.statusBar().showMessage(message)

    def _save_run(self, status: RunStatus, error: str | None = None) -> None:
        """Finalize the active run using its idempotent artifact writer.

        Args:
            status (RunStatus): Final run status such as completed, failed, or
                cancelled.
            error (str | None): Optional failure detail stored with the run.
        """
        if self.session is None or self.simulation is None:
            return
        try:
            artifacts = self.session.finalize(
                self.simulation, status=status, error=error
            )
        except OSError as save_error:
            self.statusBar().showMessage(f"Could not save run: {save_error}")
        else:
            if status == "completed":
                self.statusBar().showMessage(
                    f"Saved {artifacts.output_path} and "
                    f"{artifacts.record_path}"
                )

    def _reset_simulation(self) -> None:
        """Recreate the current fleet from its loaded map without file I/O."""
        if self.loaded_config is not None and self.loaded_map_path is not None:
            self._load_config(
                self.loaded_config, self.loaded_map_path, self._source_bytes
            )

    def _refresh_simulation_stats(self) -> None:
        """Show fleet counts and progress for the current simulation state."""
        simulation = self.simulation
        if simulation is None:
            return
        delivered = sum(drone.delivered for drone in simulation.drones)
        in_flight = sum(
            drone.in_transit is not None for drone in simulation.drones
        )
        total = len(simulation.drones)
        self.turn_label.setText(f"Turn       {simulation.turn}")
        self.drone_label.setText(f"Fleet      {total}")
        self.delivered_label.setText(f"Delivered  {delivered}")
        self.in_flight_label.setText(f"In flight   {in_flight}")
        self.waiting_label.setText(f"At hubs     {total-delivered-in_flight}")
        self.progress.setValue(round(100 * delivered / total))
        self.progress.setFormat(f"{delivered} / {total} delivered (%p%)")

    def _capture_display_state(self) -> None:
        """Copy completed-turn occupancy and traffic for the inspector."""
        current = self.simulation
        if current is None or self.loaded_config is None:
            self._display_simulation = None
            return
        # The view copy shares only the unused solver and immutable specs.
        # Occupancy and traffic stay independent while a turn interpolates.
        snapshot = Simulation.from_config(self.loaded_config, current.solver)
        snapshot.turn = current.turn
        for original, copy in zip(current.drones, snapshot.drones):
            copy.current_zone = snapshot.network.zones[
                original.current_zone.name
            ]
            copy.delivered = original.delivered
            copy.transit_remaining = original.transit_remaining
            if original.in_transit is not None:
                copy.in_transit = snapshot.network.connections[
                    original.in_transit.identifier()
                ]
        for name, connection in snapshot.network.connections.items():
            connection.current_traffic = (
                current.network.connections[name].current_traffic
            )
        self._display_simulation = snapshot
        self.inspector.refresh(snapshot)

    def _object_selected(self, object_type: str, name: str) -> None:
        """Pin a canvas object and update the inspection prompt.

        Args:
            object_type (str): Stored item kind, such as hub or connection.
            name (str): Stable zone or connection identifier.
        """
        self.inspector.select_object(object_type, name)
        self.inspector.refresh(self._display_simulation)
        self.selection_detail.setText(
            f"Inspecting {name}" if name else
            "Click a hub or connection to pin its live details."
        )

    def map_view_fit(self) -> None:
        """Fit all map items without changing their aspect ratio."""
        self.canvas.fit_map()

    def show_inspector(self) -> None:
        """Show the inspector and refresh its pinned completed-turn details."""
        self.inspector.show()
        self.inspector.raise_()
        self.inspector.activateWindow()
        self.inspector.refresh(self._display_simulation)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Cancel work and wait for worker threads before closing the window.

        Args:
            event (QCloseEvent): Close event passed by Qt.
        """
        self._closed = True
        self._discard_session()
        for worker in self._workers:
            worker.wait()
        super().closeEvent(event)


class DesktopApplication:
    """Desktop application owner for QApplication and its main window.

    Attributes:
        argv (list[str]): Argument vector used to create QApplication.
        map_root (Path | None): Optional map library directory.
        initial_map (Path | None): Optional map file loaded on startup.
        solver_name (str): Solver registry name passed to each run.
        output_dir (Path | None): Optional generated-run directory.
        records_dir (Path | None): Optional solver-record directory.
        solver_parameters (dict[str, float | int]): Numeric solver options.
    """

    def __init__(
        self, map_root: Path | None = None,
        initial_map: Path | None = None, solver_name: str = DEFAULT_SOLVER,
        output_dir: Path | None = None, records_dir: Path | None = None,
        solver_parameters: dict[str, float | int] | None = None,
        argv: list[str] | None = None,
    ) -> None:
        """Capture desktop launch options without initializing Qt.

        Args:
            map_root (Path | None): Optional map library directory.
            initial_map (Path | None): Optional map file to load on launch.
            solver_name (str): Solver registry name for the run.
            output_dir (Path | None): Optional output directory for run
                artifacts.
            records_dir (Path | None): Optional directory for solver records.
            solver_parameters (dict[str, float | int] | None): Optional numeric
                solver settings.
            argv (list[str] | None): Qt command-line arguments; defaults to
                ``sys.argv``.
        """
        self.argv = list(sys.argv if argv is None else argv)
        self.map_root = map_root
        self.initial_map = initial_map
        self.solver_name = solver_name
        self.output_dir = output_dir
        self.records_dir = records_dir
        self.solver_parameters = dict(solver_parameters or {})

    def run(self) -> int:
        """Launch the map library and run the Qt event loop.

        Returns:
            int: Exit status returned by the Qt application event loop.
        """
        app = QApplication.instance() or QApplication(self.argv)
        app.setApplicationName("Fly-In Drones")
        window = MainWindow(
            self.map_root, self.initial_map, self.solver_name,
            self.output_dir, self.records_dir,
            solver_parameters=self.solver_parameters,
        )
        window.show()
        return app.exec()


def main() -> None:
    """Launch the desktop application and exit with its status."""
    sys.exit(DesktopApplication().run())
