"""
Interactive 2D frame designer panel.

Click on the canvas to place nodes, click two nodes to connect them into a
beam/column element, assign section properties, supports (Fixed / Pinned /
Roller-X / Roller-Y), nodal point loads, UDLs, point loads and linearly
varying (trapezoidal) loads on each element, then solve for displacements,
reactions and member end forces and view the deformed shape.

This module exposes FrameDesignerPanel, a ttk.Frame that can be embedded in a
notebook (see eme302_suite.py) or run standalone with `python frame_designer.py`.

Requires: numpy, matplotlib (tkinter ships with standard Python on Windows/Mac;
on Linux install e.g. `sudo apt install python3-tk`).
"""

import json
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from fe_engine import Node, Element, Structure, LOAD_DIRS, LOAD_DIR_LABELS
from element_report import element_report, global_report

# Support types as shown in the UI. The two roller entries map onto the
# engine's ("Roller", roller_axis) pair.
SUPPORT_TYPES = ["Free", "Pinned", "Roller-Y", "Roller-X", "Fixed"]

# Load direction: human label <-> engine tag
DIR_LABELS = [LOAD_DIR_LABELS[d] for d in LOAD_DIRS]
LABEL_TO_DIR = {LOAD_DIR_LABELS[d]: d for d in LOAD_DIRS}


def dir_label(tag):
    return LOAD_DIR_LABELS.get(tag, LOAD_DIR_LABELS["local"])


def dir_tag(label):
    return LABEL_TO_DIR.get(label, "local")


# How the global DOFs are named in the matrices and reports
DOF_STYLE_LABELS = {"uvt": "u / v / th",
                    "FxFyM": "Fx / Fy / M",
                    "q": "q1, q2, q3 ..."}
LABEL_TO_DOF_STYLE = {v: k for k, v in DOF_STYLE_LABELS.items()}
GRID_SNAP = 0.25  # metres, snap tolerance when clicking
DEFAULT_VIEW_HALF_RANGE = 10.0  # metres; fixed canvas view, does not autoscale
DEFORMED_NPTS = 41  # samples per member when drawing the deflected curve


def split_support(ui_support):
    """UI support name -> (engine support, roller_axis)."""
    if ui_support == "Roller-X":
        return "Roller", "X"
    if ui_support == "Roller-Y":
        return "Roller", "Y"
    return ui_support, "Y"


# ----------------------------------------------------------------------
# Data model wrapper (keeps GUI-friendly extra fields alongside fe_engine)
# ----------------------------------------------------------------------
class NodeItem:
    _next_id = 0

    def __init__(self, x, y, name=None):
        self.id = NodeItem._next_id
        NodeItem._next_id += 1
        self.x = x
        self.y = y
        # Display name only. A node's place in the global Q vector comes from
        # its position in the Nodes list, not from this or from the id.
        self.name = name or f"N{self.id}"
        # hand-written names for this node's three DOFs; blank = generated
        self.dof_names = ["", "", ""]
        self.support = "Free"          # one of SUPPORT_TYPES (translations)
        self.restrain_rotation = False  # chosen independently of the above
        self.Fx = 0.0
        self.Fy = 0.0
        self.M = 0.0

    def label(self):
        return self.name

    def restraints(self):
        """(x_held, y_held, rotation_held) actually applied at this node."""
        if self.support in ("Fixed", "Pinned"):
            tx, ty = True, True
        elif self.support == "Roller-X":
            tx, ty = True, False
        elif self.support == "Roller-Y":
            tx, ty = False, True
        else:
            tx, ty = False, False
        return tx, ty, bool(self.restrain_rotation)

    def is_supported(self):
        return any(self.restraints())

    def support_text(self):
        """Short description for lists, e.g. 'Roller-Y, no rot'."""
        tx, ty, rot = self.restraints()
        if tx and ty and rot:
            return "Fixed"
        if not (tx or ty):
            return "Rotation only" if rot else "Free"
        return self.support + (", no rot" if rot else "")

    def to_dict(self):
        return {"x": self.x, "y": self.y, "name": self.name,
                "dof_names": list(self.dof_names),
                "support": self.support,
                "restrain_rotation": self.restrain_rotation,
                "Fx": self.Fx, "Fy": self.Fy, "M": self.M}


class ElementItem:
    _next_id = 0

    def __init__(self, node_i, node_j):
        self.id = ElementItem._next_id
        ElementItem._next_id += 1
        self.ni = node_i
        self.nj = node_j
        self.E = 200e9
        self.A = 5e-4
        self.I = 1e-5
        self.udl = 0.0          # N/m, uniform, downward negative
        self.w1 = 0.0           # trapezoidal load intensity at node i
        self.w2 = 0.0           # trapezoidal load intensity at node j
        self.udl_dir = "local"  # "local" | "gx" | "gy"
        self.lvl_dir = "local"
        self.release_i = False  # hinge at the node i end (no moment carried)
        self.release_j = False  # hinge at the node j end
        # hand-written names for the extra rotation DOFs a hinged end gets
        self.release_names = ["", ""]
        self.point_loads = []   # list of [P, a, dir]  (P in N, a in m from node i)

    def label(self):
        return f"E{self.id}: {self.ni.label()}-{self.nj.label()}"

    def length(self):
        return np.hypot(self.nj.x - self.ni.x, self.nj.y - self.ni.y)

    def end_marks(self):
        """Short suffix showing which ends are hinged, for list displays."""
        if self.release_i and self.release_j:
            return " (o-o)"
        if self.release_i:
            return " (o-)"
        if self.release_j:
            return " (-o)"
        return ""

    def to_dict(self, node_index):
        return {"ni": node_index[self.ni], "nj": node_index[self.nj],
                "E": self.E, "A": self.A, "I": self.I,
                "udl": self.udl, "w1": self.w1, "w2": self.w2,
                "udl_dir": self.udl_dir, "lvl_dir": self.lvl_dir,
                "release_i": self.release_i, "release_j": self.release_j,
                "release_names": list(self.release_names),
                "point_loads": [[float(p[0]), float(p[1]),
                                 p[2] if len(p) > 2 else "local"]
                                for p in self.point_loads]}


# ----------------------------------------------------------------------
# Main panel
# ----------------------------------------------------------------------
class FrameDesignerPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.nodes = []
        self.elements = []
        self.mode = tk.StringVar(value="node")   # node | element | select
        self.pending_node_for_element = None
        self.selected_node = None
        self.selected_element = None

        self.result = None       # last solve() result dict
        self.struct = None       # fe_engine Structure used for that solve
        self.fe_elements = {}    # ElementItem -> fe_engine Element
        self.elem_tabs = {}      # ElementItem -> (tab frame, text widget)
        self.deformed_scale = tk.DoubleVar(value=50.0)
        self.show_deformed = tk.BooleanVar(value=False)

        # Fixed (manual) view window -- never autoscales to content
        self.view_xmin = tk.DoubleVar(value=-DEFAULT_VIEW_HALF_RANGE)
        self.view_xmax = tk.DoubleVar(value=DEFAULT_VIEW_HALF_RANGE)
        self.view_ymin = tk.DoubleVar(value=-DEFAULT_VIEW_HALF_RANGE)
        self.view_ymax = tk.DoubleVar(value=DEFAULT_VIEW_HALF_RANGE)

        self._build_layout()
        self._redraw()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)

        # Left: canvas
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        toolbar = ttk.Frame(left)
        toolbar.pack(side="top", fill="x", padx=4, pady=4)

        ttk.Label(toolbar, text="Mode:").pack(side="left", padx=(0, 4))
        ttk.Radiobutton(toolbar, text="Add Node", variable=self.mode,
                        value="node", command=self._reset_pending).pack(side="left")
        ttk.Radiobutton(toolbar, text="Add Element (click 2 nodes)",
                        variable=self.mode, value="element",
                        command=self._reset_pending).pack(side="left")
        ttk.Radiobutton(toolbar, text="Select / Edit", variable=self.mode,
                        value="select", command=self._reset_pending).pack(side="left")

        ttk.Button(toolbar, text="Node by Coords...",
                   command=self._add_node_by_coords_dialog).pack(side="left", padx=10)
        ttk.Button(toolbar, text="Delete Selected",
                   command=self._delete_selected).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Clear All",
                   command=self._clear_all).pack(side="left", padx=4)

        ttk.Checkbutton(toolbar, text="Show deformed shape",
                        variable=self.show_deformed,
                        command=self._redraw).pack(side="right", padx=4)
        ttk.Button(toolbar, text="Auto", width=5,
                   command=self._auto_scale).pack(side="right", padx=(0, 4))
        ttk.Spinbox(toolbar, from_=1, to=5000, increment=10, width=6,
                    textvariable=self.deformed_scale,
                    command=self._redraw).pack(side="right", padx=4)
        ttk.Label(toolbar, text="Scale:").pack(side="right")

        # View-range toolbar (fixed, manual -- canvas never autoscales to content)
        view_bar = ttk.Frame(left)
        view_bar.pack(side="top", fill="x", padx=4)
        ttk.Label(view_bar, text="View range:  X:").pack(side="left")
        ttk.Entry(view_bar, textvariable=self.view_xmin, width=6).pack(side="left")
        ttk.Label(view_bar, text="to").pack(side="left", padx=2)
        ttk.Entry(view_bar, textvariable=self.view_xmax, width=6).pack(side="left")
        ttk.Label(view_bar, text="  Y:").pack(side="left", padx=(8, 0))
        ttk.Entry(view_bar, textvariable=self.view_ymin, width=6).pack(side="left")
        ttk.Label(view_bar, text="to").pack(side="left", padx=2)
        ttk.Entry(view_bar, textvariable=self.view_ymax, width=6).pack(side="left")
        ttk.Button(view_bar, text="Apply View",
                   command=self._redraw).pack(side="left", padx=6)
        ttk.Button(view_bar, text="Fit to Structure",
                   command=self._fit_view_to_structure).pack(side="left", padx=2)
        ttk.Button(view_bar, text="Save Model...",
                   command=self.save_model).pack(side="right", padx=2)
        ttk.Button(view_bar, text="Load Model...",
                   command=self.load_model_dialog).pack(side="right", padx=2)

        self.fig = Figure(figsize=(8, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        solve_bar = ttk.Frame(left)
        solve_bar.pack(side="bottom", fill="x", padx=4, pady=6)
        style = ttk.Style()
        style.configure("Solve.TButton", font=("TkDefaultFont", 11, "bold"))
        ttk.Button(solve_bar, text="SOLVE STRUCTURE", command=self.solve,
                   style="Solve.TButton").pack(side="left", fill="x", expand=True)

        # Right: property / results / per-element working panel. Wide enough
        # for a 6x6 matrix at 9pt Courier; the panes also scroll sideways.
        right = ttk.Frame(main, width=620)
        right.pack(side="right", fill="both")
        right.pack_propagate(False)

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)

        self.tab_props = ttk.Frame(self.tabs)
        self.tab_results = ttk.Frame(self.tabs)
        self.tab_dofs = ttk.Frame(self.tabs)
        self.tab_global = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_props, text="Properties")
        self.tabs.add(self.tab_results, text="Results")
        self.tabs.add(self.tab_dofs, text="DOF Names")
        self.tabs.add(self.tab_global, text="Global Q / KG")

        self._build_props_tab()
        self._build_results_tab()
        self._build_dof_tab()
        self.global_text, _ = self._make_scrolled_text(self.tab_global)
        self._set_text(self.global_text,
                       "Add some elements to see the assembled global system.")

    def _build_props_tab(self):
        # Wrap all Properties-tab content in a scrollable canvas, since the
        # node/element editors can be taller than the fixed-width side panel.
        outer = self.tab_props
        scroll_canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=vsb.set)
        scroll_canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        frame = ttk.Frame(scroll_canvas)
        frame_window = scroll_canvas.create_window((0, 0), window=frame, anchor="nw")

        def _on_frame_configure(event):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        def _on_canvas_configure(event):
            # Keep the inner frame's width matched to the canvas so widgets
            # don't have to be individually re-wrapped as it resizes.
            scroll_canvas.itemconfig(frame_window, width=event.width)

        frame.bind("<Configure>", _on_frame_configure)
        scroll_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            # Windows/Mac send delta in multiples of 120; Linux sends Button-4/5.
            if event.num == 4:
                scroll_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                scroll_canvas.yview_scroll(1, "units")
            else:
                scroll_canvas.yview_scroll(int(-event.delta / 120), "units")

        def _bind_wheel(_event):
            scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            scroll_canvas.bind_all("<Button-4>", _on_mousewheel)
            scroll_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_wheel(_event):
            scroll_canvas.unbind_all("<MouseWheel>")
            scroll_canvas.unbind_all("<Button-4>")
            scroll_canvas.unbind_all("<Button-5>")

        # Only capture the scroll wheel while the pointer is actually over
        # this panel, so it doesn't hijack scrolling elsewhere.
        scroll_canvas.bind("<Enter>", _bind_wheel)
        scroll_canvas.bind("<Leave>", _unbind_wheel)

        lst_frame = ttk.LabelFrame(frame, text="Nodes  (this order is the order of Q)")
        lst_frame.pack(fill="both", padx=6, pady=4)
        self.node_list = tk.Listbox(lst_frame, height=6)
        self.node_list.pack(fill="x", padx=4, pady=4)
        self.node_list.bind("<<ListboxSelect>>", self._on_node_list_select)
        order_bar = ttk.Frame(lst_frame)
        order_bar.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(order_bar, text="Move Up",
                   command=lambda: self._move_node(-1)).pack(side="left")
        ttk.Button(order_bar, text="Move Down",
                   command=lambda: self._move_node(1)).pack(side="left", padx=4)
        ttk.Label(order_bar, text="   DOF labels:").pack(side="left")
        self.dof_style = tk.StringVar(value=DOF_STYLE_LABELS["uvt"])
        cb = ttk.Combobox(order_bar, textvariable=self.dof_style, state="readonly",
                          width=14, values=list(DOF_STYLE_LABELS.values()))
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_reports())

        elem_frame = ttk.LabelFrame(frame, text="Elements")
        elem_frame.pack(fill="both", padx=6, pady=4)
        self.elem_list = tk.Listbox(elem_frame, height=6)
        self.elem_list.pack(fill="x", padx=4, pady=4)
        self.elem_list.bind("<<ListboxSelect>>", self._on_elem_list_select)

        # Node editor
        self.node_editor = ttk.LabelFrame(frame, text="Node Editor")
        self.node_editor.pack(fill="x", padx=6, pady=6)
        self._build_node_editor(self.node_editor)

        # Element editor
        self.elem_editor = ttk.LabelFrame(frame, text="Element Editor")
        self.elem_editor.pack(fill="x", padx=6, pady=6)
        self._build_elem_editor(self.elem_editor)

    def _build_node_editor(self, parent):
        self.node_editor_target = None
        row = 0
        self.n_coord_lbl = ttk.Label(parent, text="Node: -")
        self.n_coord_lbl.grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        row += 1

        ttk.Label(parent, text="Name:").grid(row=row, column=0, sticky="w", padx=4)
        self.n_name = tk.StringVar(value="")
        ttk.Entry(parent, textvariable=self.n_name, width=14).grid(
            row=row, column=1, padx=4, pady=2)
        row += 1
        ttk.Label(parent,
                  text="(Used everywhere this node is shown, and in\n"
                       " the global DOF labels u_<name>, v_<name>,\n"
                       " th_<name>. The node's POSITION in the Nodes\n"
                       " list, not its name, sets where it sits in Q.)",
                  font=("TkDefaultFont", 7), foreground="gray").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4)
        row += 1

        ttk.Label(parent, text="X (m):").grid(row=row, column=0, sticky="w", padx=4)
        self.n_x = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.n_x, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1

        ttk.Label(parent, text="Y (m):").grid(row=row, column=0, sticky="w", padx=4)
        self.n_y = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.n_y, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1

        ttk.Label(parent, text="Support:").grid(row=row, column=0, sticky="w", padx=4)
        self.n_support = tk.StringVar(value="Free")
        cb = ttk.Combobox(parent, textvariable=self.n_support, values=SUPPORT_TYPES,
                          state="readonly", width=10)
        cb.grid(row=row, column=1, sticky="w", padx=4, pady=2)
        cb.bind("<<ComboboxSelected>>", self._on_support_changed)
        row += 1
        ttk.Label(parent,
                  text="(Chooses which TRANSLATIONS are held.\n"
                       " Roller-Y restrains global Y -- rolls\n"
                       " horizontally.  Roller-X restrains global\n"
                       " X -- rolls vertically.)",
                  font=("TkDefaultFont", 7), foreground="gray").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4)
        row += 1

        self.n_rot = tk.BooleanVar(value=False)
        self.n_rot_check = ttk.Checkbutton(
            parent, text="Restrain rotation at this node", variable=self.n_rot)
        self.n_rot_check.grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 0))
        row += 1
        ttk.Label(parent,
                  text="(Rotation is chosen separately from the\n"
                       " translations. Leave it clear for an ordinary\n"
                       " pinned or roller support. Tick it on a roller\n"
                       " to get a GUIDED support: it still slides along\n"
                       " its free axis but cannot rotate, so it carries\n"
                       " a reaction moment. Tick it on a Free node to\n"
                       " restrain rotation alone -- a symmetry line.\n"
                       " Fixed always holds rotation.)",
                  font=("TkDefaultFont", 7), foreground="gray").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4)
        row += 1

        ttk.Label(parent, text="Fx (N):").grid(row=row, column=0, sticky="w", padx=4)
        self.n_fx = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.n_fx, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1

        ttk.Label(parent, text="Fy (N):").grid(row=row, column=0, sticky="w", padx=4)
        self.n_fy = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.n_fy, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1

        ttk.Label(parent, text="M (Nm):").grid(row=row, column=0, sticky="w", padx=4)
        self.n_m = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.n_m, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1

        ttk.Button(parent, text="Apply to Selected Node",
                   command=self._apply_node_props).grid(row=row, column=0, columnspan=2,
                                                        sticky="ew", padx=4, pady=6)

    def _build_elem_editor(self, parent):
        self.elem_editor_target = None
        row = 0
        self.e_lbl = ttk.Label(parent, text="Element: -")
        self.e_lbl.grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        row += 1

        ttk.Label(parent, text="Node i position:", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 0))
        row += 1
        ttk.Label(parent, text="  X (m):").grid(row=row, column=0, sticky="w", padx=4)
        self.e_ni_x = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.e_ni_x, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1
        ttk.Label(parent, text="  Y (m):").grid(row=row, column=0, sticky="w", padx=4)
        self.e_ni_y = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.e_ni_y, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1

        ttk.Label(parent, text="Node j position:", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 0))
        row += 1
        ttk.Label(parent, text="  X (m):").grid(row=row, column=0, sticky="w", padx=4)
        self.e_nj_x = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.e_nj_x, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1
        ttk.Label(parent, text="  Y (m):").grid(row=row, column=0, sticky="w", padx=4)
        self.e_nj_y = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.e_nj_y, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1
        ttk.Label(parent, text="(node j can also be set below by\n length + angle instead of X/Y)",
                  font=("TkDefaultFont", 7), foreground="gray").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4)
        row += 1
        ttk.Label(parent, text="(moving node i keeps node j in place\n unless length/angle also set)",
                  font=("TkDefaultFont", 7), foreground="gray").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2,
                                                        sticky="ew", pady=4)
        row += 1

        ttk.Label(parent, text="Length (m):").grid(row=row, column=0, sticky="w", padx=4)
        self.e_length = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.e_length, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1
        ttk.Label(parent, text="(edits length by moving node j\n along the current direction)",
                  font=("TkDefaultFont", 7), foreground="gray").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4)
        row += 1

        ttk.Label(parent, text="Angle (deg, from +X axis):").grid(row=row, column=0, sticky="w", padx=4)
        self.e_angle = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.e_angle, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1
        ttk.Label(parent, text="(edits angle by rotating node j\n about node i, length preserved)",
                  font=("TkDefaultFont", 7), foreground="gray").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4)
        row += 1
        ttk.Label(parent, text="Length/Angle fields take priority over\n"
                              "node j X/Y if both are changed at once.",
                  font=("TkDefaultFont", 7), foreground="gray").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2,
                                                        sticky="ew", pady=4)
        row += 1

        ttk.Label(parent, text="E (Pa):").grid(row=row, column=0, sticky="w", padx=4)
        self.e_E = tk.StringVar(value="200e9")
        ttk.Entry(parent, textvariable=self.e_E, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1

        ttk.Label(parent, text="A (m^2):").grid(row=row, column=0, sticky="w", padx=4)
        self.e_A = tk.StringVar(value="5e-4")
        ttk.Entry(parent, textvariable=self.e_A, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1

        ttk.Label(parent, text="I (m^4):").grid(row=row, column=0, sticky="w", padx=4)
        self.e_I = tk.StringVar(value="1e-5")
        ttk.Entry(parent, textvariable=self.e_I, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2,
                                                        sticky="ew", pady=4)
        row += 1

        ttk.Label(parent, text="End connections:",
                  font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(2, 0))
        row += 1
        self.e_rel_i = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Hinge at node i end", variable=self.e_rel_i).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8)
        row += 1
        self.e_rel_j = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Hinge at node j end", variable=self.e_rel_j).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=8)
        row += 1
        ttk.Label(parent,
                  text="(Unticked = rigidly connected to whatever\n"
                       " else meets that joint. Ticked = this member\n"
                       " carries no moment there and rotates freely,\n"
                       " while other members at the joint stay rigid.\n"
                       " The joint itself is not tied to ground -- that\n"
                       " is what the node Support setting does.)",
                  font=("TkDefaultFont", 7), foreground="gray").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2,
                                                        sticky="ew", pady=4)
        row += 1

        ttk.Label(parent, text="UDL (N/m, - = down):").grid(row=row, column=0, sticky="w", padx=4)
        self.e_udl = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.e_udl, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1
        ttk.Label(parent, text="   acting:").grid(row=row, column=0, sticky="w", padx=4)
        self.e_udl_dir = tk.StringVar(value=dir_label("local"))
        ttk.Combobox(parent, textvariable=self.e_udl_dir, values=DIR_LABELS,
                     state="readonly", width=16).grid(row=row, column=1, sticky="w", padx=4, pady=2)
        row += 1

        ttk.Label(parent, text="Trapezoid w1 @ node i (N/m):").grid(row=row, column=0, sticky="w", padx=4)
        self.e_w1 = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.e_w1, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1

        ttk.Label(parent, text="Trapezoid w2 @ node j (N/m):").grid(row=row, column=0, sticky="w", padx=4)
        self.e_w2 = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.e_w2, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1
        ttk.Label(parent, text="   acting:").grid(row=row, column=0, sticky="w", padx=4)
        self.e_lvl_dir = tk.StringVar(value=dir_label("local"))
        ttk.Combobox(parent, textvariable=self.e_lvl_dir, values=DIR_LABELS,
                     state="readonly", width=16).grid(row=row, column=1, sticky="w", padx=4, pady=2)
        row += 1
        ttk.Label(parent,
                  text="(Intensity is per metre OF MEMBER. For a load\n"
                       " given per metre of horizontal plan, multiply\n"
                       " it by cos(member angle) first.)",
                  font=("TkDefaultFont", 7), foreground="gray").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=2,
                                                        sticky="ew", pady=4)
        row += 1

        ttk.Label(parent, text="Point loads (P, a from node i):").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4)
        row += 1
        self.pl_list = tk.Listbox(parent, height=3)
        self.pl_list.grid(row=row, column=0, columnspan=2, sticky="ew", padx=4)
        row += 1

        pl_add_frame = ttk.Frame(parent)
        pl_add_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        ttk.Label(pl_add_frame, text="P(N):").pack(side="left")
        self.pl_P = tk.StringVar(value="0")
        ttk.Entry(pl_add_frame, textvariable=self.pl_P, width=8).pack(side="left", padx=2)
        ttk.Label(pl_add_frame, text="a(m):").pack(side="left")
        self.pl_a = tk.StringVar(value="0")
        ttk.Entry(pl_add_frame, textvariable=self.pl_a, width=8).pack(side="left", padx=2)
        row += 1

        pl_dir_frame = ttk.Frame(parent)
        pl_dir_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        self.pl_dir = tk.StringVar(value=dir_label("local"))
        ttk.Combobox(pl_dir_frame, textvariable=self.pl_dir, values=DIR_LABELS,
                     state="readonly", width=16).pack(side="left")
        ttk.Button(pl_dir_frame, text="Add", command=self._add_point_load).pack(side="left", padx=4)
        ttk.Button(pl_dir_frame, text="Remove Sel.",
                   command=self._remove_point_load).pack(side="left")
        row += 1

        ttk.Button(parent, text="Apply to Selected Element",
                   command=self._apply_elem_props).grid(row=row, column=0, columnspan=2,
                                                        sticky="ew", padx=4, pady=6)

    def _build_results_tab(self):
        frame = self.tab_results
        btns = ttk.Frame(frame)
        btns.pack(side="bottom", fill="x", padx=6, pady=(0, 6))
        ttk.Button(btns, text="Save full report (results + all element working)...",
                   command=self.save_report).pack(fill="x")
        self.results_text, _ = self._make_scrolled_text(frame)
        self._set_text(self.results_text, "Click SOLVE STRUCTURE to see results here.")

    @staticmethod
    def _make_scrolled_text(parent):
        """Read-only monospace pane with both scrollbars."""
        wrap = ttk.Frame(parent)
        wrap.pack(side="top", fill="both", expand=True, padx=6, pady=6)
        txt = tk.Text(wrap, wrap="none", font=("Courier New", 9))
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=txt.yview)
        hsb = ttk.Scrollbar(wrap, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set, state="disabled")
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        txt.pack(side="left", fill="both", expand=True)
        return txt, wrap

    @staticmethod
    def _set_text(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    # ------------------------------------------------------------------
    # DOF naming
    # ------------------------------------------------------------------
    def _build_dof_tab(self):
        frame = self.tab_dofs
        ttk.Label(frame, wraplength=560, justify="left",
                  text="Name each degree of freedom yourself. Whatever you type here is "
                       "printed next to that row in the Q vector, KG, the reduced K_ff / "
                       "Q_f, the reactions, and next to the local and global force "
                       "vectors on every element tab.\n\n"
                       "Leave a box empty to fall back to the generated label. Names "
                       "belong to the node, so reordering the nodes carries each name to "
                       "its new row.").pack(fill="x", padx=8, pady=(8, 4))

        btns = ttk.Frame(frame)
        btns.pack(fill="x", padx=8)
        ttk.Button(btns, text="Number the FREE DOFs q1, q2, ...",
                   command=lambda: self._autofill_dofs(free_only=True)).pack(side="left")
        ttk.Button(btns, text="Number ALL DOFs q1, q2, ...",
                   command=lambda: self._autofill_dofs(free_only=False)).pack(side="left",
                                                                              padx=4)
        ttk.Button(btns, text="Clear all",
                   command=self._clear_dof_names).pack(side="left")
        ttk.Label(frame, wraplength=560, justify="left", foreground="gray",
                  font=("TkDefaultFont", 8),
                  text="'Number the FREE DOFs' matches the lab worksheets: the restrained "
                       "DOFs drop out of the reduced system, so only the ones that survive "
                       "into the displacement vector get a q number.").pack(
            fill="x", padx=8, pady=(2, 6))

        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True, padx=8)
        self.dof_canvas = tk.Canvas(body, highlightthickness=0)
        vsb = ttk.Scrollbar(body, orient="vertical", command=self.dof_canvas.yview)
        self.dof_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.dof_canvas.pack(side="left", fill="both", expand=True)
        self.dof_inner = ttk.Frame(self.dof_canvas)
        self.dof_canvas.create_window((0, 0), window=self.dof_inner, anchor="nw")
        self.dof_inner.bind(
            "<Configure>",
            lambda e: self.dof_canvas.configure(scrollregion=self.dof_canvas.bbox("all")))
        self.dof_canvas.bind(
            "<Enter>",
            lambda e: self.dof_canvas.bind_all(
                "<MouseWheel>",
                lambda ev: self.dof_canvas.yview_scroll(int(-ev.delta / 120), "units")))
        self.dof_canvas.bind("<Leave>",
                             lambda e: self.dof_canvas.unbind_all("<MouseWheel>"))

        bar = ttk.Frame(frame)
        bar.pack(fill="x", padx=8, pady=8)
        ttk.Button(bar, text="Apply names", command=self._apply_dof_names).pack(
            side="left", fill="x", expand=True)

        self.dof_rows = []        # list of (owner, slot, StringVar)
        self.dof_signature = None

    def _dof_signature(self, struct):
        """What the table depends on: which DOFs exist and who owns them."""
        if struct is None:
            return None
        return (tuple(id(n) for n in self.nodes),
                tuple(sorted(struct.release_dofs.values())),
                struct.ndof)

    def _sync_dof_tab(self, struct):
        """Rebuild the name table only when the set of DOFs actually changes."""
        sig = self._dof_signature(struct)
        if sig == self.dof_signature:
            return
        self.dof_signature = sig
        for w in self.dof_inner.winfo_children():
            w.destroy()
        self.dof_rows = []
        if struct is None:
            ttk.Label(self.dof_inner,
                      text="Add some elements first.").grid(row=0, column=0, padx=4, pady=4)
            return

        auto = struct.dof_labels(self.dof_style_tag())
        restrained = set(struct.restrained_global_dofs())
        for col, head in enumerate(["Q row", "belongs to", "status", "your name"]):
            ttk.Label(self.dof_inner, text=head,
                      font=("TkDefaultFont", 9, "bold")).grid(
                row=0, column=col, padx=4, pady=3, sticky="w")

        # map each global DOF back to the GUI object that stores its name
        owner_of = {}
        for k, n in enumerate(self.nodes):
            for c in range(3):
                owner_of[3*k + c] = (n, c)
        for (k, end), dof in struct.release_dofs.items():
            owner_of[dof] = (self.elements[k], 3 + (0 if end == "i" else 1))

        for dof in range(struct.ndof):
            who, _nd = struct.dof_owner(dof)
            state = "restrained" if dof in restrained else "free"
            ttk.Label(self.dof_inner, text=str(dof)).grid(
                row=dof + 1, column=0, padx=4, sticky="w")
            ttk.Label(self.dof_inner, text=who).grid(
                row=dof + 1, column=1, padx=4, sticky="w")
            ttk.Label(self.dof_inner, text=state,
                      foreground="gray" if state == "free" else "firebrick").grid(
                row=dof + 1, column=2, padx=4, sticky="w")
            owner, slot = owner_of[dof]
            current = (owner.dof_names[slot] if slot < 3
                       else owner.release_names[slot - 3])
            var = tk.StringVar(value=current)
            ent = ttk.Entry(self.dof_inner, textvariable=var, width=12)
            ent.grid(row=dof + 1, column=3, padx=4, pady=1)
            ent.bind("<Return>", lambda e: self._apply_dof_names())
            ttk.Label(self.dof_inner, text=f"(else {auto[dof]})",
                      foreground="gray", font=("TkDefaultFont", 8)).grid(
                row=dof + 1, column=4, padx=4, sticky="w")
            self.dof_rows.append((owner, slot, var))

    def _autofill_dofs(self, free_only):
        struct, _mapping, _result = self._report_context()
        if struct is None:
            return
        restrained = set(struct.restrained_global_dofs())
        k = 0
        for dof, (_owner, _slot, var) in enumerate(self.dof_rows):
            if free_only and dof in restrained:
                var.set("")
                continue
            k += 1
            var.set(f"q{k}")
        self._apply_dof_names()

    def _clear_dof_names(self):
        for _owner, _slot, var in self.dof_rows:
            var.set("")
        self._apply_dof_names()

    def _apply_dof_names(self):
        for owner, slot, var in self.dof_rows:
            name = var.get().strip()
            if slot < 3:
                owner.dof_names[slot] = name
            else:
                owner.release_names[slot - 3] = name
        # A solved model keeps its own copy of the engine objects, built when
        # SOLVE was pressed. Names are labels only and change no numbers, so
        # push them straight across rather than forcing a re-solve.
        if self.struct is not None:
            for item, fen in zip(self.nodes, self.struct.nodes):
                fen.dof_names = list(item.dof_names)
            for item, fee in self.fe_elements.items():
                fee.release_names = list(item.release_names)
        self._refresh_reports()
        self._display_results_if_solved()

    def _display_results_if_solved(self):
        if self.result is not None and self.struct is not None:
            self._display_results()

    # ------------------------------------------------------------------
    # Per-element "working" tabs
    # ------------------------------------------------------------------
    def _sync_element_tabs(self):
        """One tab per element, showing every calculation it goes through."""
        if not hasattr(self, "tabs"):
            return
        # rebuild the tab set only when the elements themselves change, so
        # the user's current tab is not thrown away on every edit
        if list(self.elem_tabs.keys()) != self.elements:
            for _e, (frame, _txt) in self.elem_tabs.items():
                try:
                    self.tabs.forget(frame)
                except tk.TclError:
                    pass
                frame.destroy()
            self.elem_tabs = {}
            for e in self.elements:
                frame = ttk.Frame(self.tabs)
                txt, _ = self._make_scrolled_text(frame)
                self.tabs.add(frame, text=f" E{e.id} ")
                self.elem_tabs[e] = (frame, txt)

        struct, mapping, result = self._report_context()
        style = self.dof_style_tag()
        self._sync_dof_tab(struct)

        # the assembled global system
        if struct is None:
            self._set_text(self.global_text,
                           "Add some elements to see the assembled global system.")
        else:
            try:
                self._set_text(self.global_text,
                               global_report(struct, result, style))
            except Exception as ex:
                self._set_text(self.global_text,
                               f"Could not assemble the global system:\n{ex}")

        if not self.elem_tabs:
            return
        for e, (_frame, txt) in self.elem_tabs.items():
            fee = mapping.get(e)
            if fee is None:
                self._set_text(txt, "This element could not be built -- check that its "
                                    "two nodes are not at the same point.")
                continue
            try:
                self._set_text(txt, element_report(fee, struct, result, style=style))
            except Exception as ex:      # never let a display error kill an edit
                self._set_text(txt, f"Could not produce the working for {e.label()}:\n{ex}")

    def save_report(self):
        top = self.winfo_toplevel()
        if not self.elements:
            messagebox.showinfo("Nothing to save", "The model is empty.", parent=top)
            return
        path = filedialog.asksaveasfilename(
            parent=top, defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
            title="Save full report")
        if not path:
            return
        struct, mapping, result = self._report_context()
        header = ["=" * 78, "FRAME ANALYSIS REPORT", "=" * 78, "",
                  self.results_as_text(), ""]
        style = self.dof_style_tag()
        body = [global_report(struct, result, style)] if struct is not None else []
        body += [element_report(mapping[e], struct, result, style=style)
                 for e in self.elements if e in mapping]
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(header) + "\n\n" + "\n\n".join(body) + "\n")
        except OSError as ex:
            messagebox.showerror("Save failed", str(ex), parent=top)

    # ------------------------------------------------------------------
    # Canvas interaction
    # ------------------------------------------------------------------
    def _reset_pending(self):
        self.pending_node_for_element = None
        self._redraw()

    def _invalidate_result(self):
        """Geometry/properties changed -- the stored solution no longer applies."""
        if self.result is not None:
            self.result = None
            self.struct = None
            self.fe_elements = {}
            self.show_deformed.set(False)
            self._set_results_text("Model changed since the last solve.\n\n"
                                   "Click SOLVE STRUCTURE again to refresh the results.")
            self._sync_element_tabs()

    def _add_node_by_coords_dialog(self):
        top = self.winfo_toplevel()
        x = simpledialog.askfloat("New Node", "X coordinate (m):", parent=top)
        if x is None:
            return
        y = simpledialog.askfloat("New Node", "Y coordinate (m):", parent=top)
        if y is None:
            return
        n = NodeItem(x, y)
        self.nodes.append(n)
        self._invalidate_result()
        self._refresh_lists()
        self._redraw()

    def _fit_view_to_structure(self):
        if not self.nodes:
            return
        xs = [n.x for n in self.nodes]
        ys = [n.y for n in self.nodes]
        pad = max(1.0, 0.15 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0))
        self.view_xmin.set(round(min(xs) - pad, 2))
        self.view_xmax.set(round(max(xs) + pad, 2))
        self.view_ymin.set(round(min(ys) - pad, 2))
        self.view_ymax.set(round(max(ys) + pad, 2))
        self._redraw()

    def _auto_scale(self, redraw=True):
        """
        Pick a magnification that puts the largest deflection at roughly a
        tenth of the view span. Without this a flexible frame's deformed
        shape flies off the canvas at the default scale, and a stiff one is
        invisible.
        """
        if self.result is None or self.struct is None:
            return
        maxd = 0.0
        for e in self.elements:
            fee = self.fe_elements.get(e)
            if fee is None:
                continue
            npts = 21
            X, Y = self.struct.deflected_shape(fee, self.result["q"],
                                               npts=npts, scale=1.0)
            t = np.linspace(0.0, 1.0, npts)
            X0 = e.ni.x + t * (e.nj.x - e.ni.x)
            Y0 = e.ni.y + t * (e.nj.y - e.ni.y)
            maxd = max(maxd, float(np.max(np.hypot(X - X0, Y - Y0))))
        if maxd <= 0:
            return
        try:
            span = max(self.view_xmax.get() - self.view_xmin.get(),
                       self.view_ymax.get() - self.view_ymin.get())
        except tk.TclError:
            return
        if span <= 0:
            return
        scale = min(max(0.10 * span / maxd, 1.0), 5000.0)
        self.deformed_scale.set(round(scale, 1))
        if redraw:
            self._redraw()

    def _snap(self, x, y):
        return round(x / GRID_SNAP) * GRID_SNAP, round(y / GRID_SNAP) * GRID_SNAP

    def _pick_tol(self, frac=0.02):
        """Hit-test tolerance scaled to the current manual view range."""
        span = max(self.view_xmax.get() - self.view_xmin.get(),
                   self.view_ymax.get() - self.view_ymin.get())
        return max(span * frac, 0.05)

    def _find_nearby_node(self, x, y, tol=None):
        if tol is None:
            tol = self._pick_tol()
        best, best_d = None, tol
        for n in self.nodes:
            d = np.hypot(n.x - x, n.y - y)
            if d < best_d:
                best, best_d = n, d
        return best

    def _find_nearby_element(self, x, y, tol=None):
        if tol is None:
            tol = self._pick_tol()
        best, best_d = None, tol
        for e in self.elements:
            d = self._point_seg_dist(x, y, e.ni.x, e.ni.y, e.nj.x, e.nj.y)
            if d < best_d:
                best, best_d = e, d
        return best

    @staticmethod
    def _point_seg_dist(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        if dx == dy == 0:
            return np.hypot(px - x1, py - y1)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        cx, cy = x1 + t * dx, y1 + t * dy
        return np.hypot(px - cx, py - cy)

    def _on_canvas_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata
        mode = self.mode.get()

        if mode == "node":
            existing = self._find_nearby_node(x, y)
            if existing is None:
                sx, sy = self._snap(x, y)
                self.nodes.append(NodeItem(sx, sy))
                self._invalidate_result()
                self._refresh_lists()
                self._redraw()

        elif mode == "element":
            n = self._find_nearby_node(x, y)
            if n is None:
                return
            if self.pending_node_for_element is None:
                self.pending_node_for_element = n
            else:
                if n is not self.pending_node_for_element:
                    self.elements.append(ElementItem(self.pending_node_for_element, n))
                    self._invalidate_result()
                    self._refresh_lists()
                self.pending_node_for_element = None
            self._redraw()

        elif mode == "select":
            n = self._find_nearby_node(x, y)
            if n is not None:
                self.selected_node = n
                self.selected_element = None
                self._select_node_in_list(n)
            else:
                e = self._find_nearby_element(x, y)
                if e is not None:
                    self.selected_element = e
                    self.selected_node = None
                    self._select_elem_in_list(e)
            self._redraw()

    # ------------------------------------------------------------------
    # List <-> selection sync
    # ------------------------------------------------------------------
    def _refresh_lists(self):
        self.node_list.delete(0, "end")
        for k, n in enumerate(self.nodes):
            self.node_list.insert(
                "end", f"{k}: {n.label()}  ({n.x:.2f}, {n.y:.2f})  [{n.support_text()}]")
        self.elem_list.delete(0, "end")
        for e in self.elements:
            self.elem_list.insert("end", f"{e.label()}  L={e.length():.2f}m{e.end_marks()}")
        self._sync_element_tabs()

    def _select_node_in_list(self, n):
        idx = self.nodes.index(n)
        self.node_list.selection_clear(0, "end")
        self.node_list.selection_set(idx)
        self._load_node_editor(n)

    def _select_elem_in_list(self, e):
        idx = self.elements.index(e)
        self.elem_list.selection_clear(0, "end")
        self.elem_list.selection_set(idx)
        self._load_elem_editor(e)

    def _on_node_list_select(self, evt):
        sel = self.node_list.curselection()
        if not sel:
            return
        n = self.nodes[sel[0]]
        self.selected_node = n
        self.selected_element = None
        self._load_node_editor(n)
        self._redraw()

    def _on_elem_list_select(self, evt):
        sel = self.elem_list.curselection()
        if not sel:
            return
        e = self.elements[sel[0]]
        self.selected_element = e
        self.selected_node = None
        self._load_elem_editor(e)
        self._redraw()

    def dof_style_tag(self):
        return LABEL_TO_DOF_STYLE.get(self.dof_style.get(), "uvt")

    def _move_node(self, delta):
        """
        Move the selected node up or down the list. The list order IS the
        global DOF order, so this is how the Q vector gets reordered.
        """
        n = self.selected_node or self.node_editor_target
        if n is None or n not in self.nodes:
            messagebox.showinfo("No node selected",
                                "Select a node first, then move it.",
                                parent=self.winfo_toplevel())
            return
        i = self.nodes.index(n)
        j = i + delta
        if not (0 <= j < len(self.nodes)):
            return
        self.nodes[i], self.nodes[j] = self.nodes[j], self.nodes[i]
        self._invalidate_result()
        self._refresh_lists()
        self._select_node_in_list(n)
        self._redraw()

    def _refresh_reports(self):
        """Re-render the working tabs, e.g. after the DOF naming style changes."""
        self._sync_element_tabs()

    def _on_support_changed(self, _evt=None):
        """A Fixed support holds rotation by definition, so lock the tick box."""
        if self.n_support.get() == "Fixed":
            self.n_rot.set(True)
            self.n_rot_check.state(["disabled"])
        else:
            self.n_rot_check.state(["!disabled"])

    def _load_node_editor(self, n):
        self.node_editor_target = n
        pos = self.nodes.index(n) if n in self.nodes else -1
        self.n_coord_lbl.config(
            text=f"Node: {n.label()}  ({n.x:.2f}, {n.y:.2f})   "
                 f"-> Q rows {3*pos}-{3*pos+2}" if pos >= 0 else f"Node: {n.label()}")
        self.n_name.set(n.name)
        self.n_x.set(f"{n.x:.4f}")
        self.n_y.set(f"{n.y:.4f}")
        self.n_support.set(n.support)
        self.n_rot.set(n.restrain_rotation)
        self._on_support_changed()
        self.n_fx.set(str(n.Fx))
        self.n_fy.set(str(n.Fy))
        self.n_m.set(str(n.M))
        self.tabs.select(self.tab_props)

    def _load_elem_editor(self, e):
        self.elem_editor_target = e
        self.e_lbl.config(text=f"Element: {e.label()}  ({e.ni.label()} -> {e.nj.label()})  "
                               f"L={e.length():.2f}m")
        self.e_ni_x.set(f"{e.ni.x:.4f}")
        self.e_ni_y.set(f"{e.ni.y:.4f}")
        self.e_nj_x.set(f"{e.nj.x:.4f}")
        self.e_nj_y.set(f"{e.nj.y:.4f}")
        self.e_length.set(f"{e.length():.4f}")
        self.e_angle.set(f"{np.degrees(np.arctan2(e.nj.y - e.ni.y, e.nj.x - e.ni.x)):.4f}")
        self.e_E.set(str(e.E))
        self.e_A.set(str(e.A))
        self.e_I.set(str(e.I))
        self.e_udl.set(str(e.udl))
        self.e_w1.set(str(e.w1))
        self.e_w2.set(str(e.w2))
        self.e_udl_dir.set(dir_label(e.udl_dir))
        self.e_lvl_dir.set(dir_label(e.lvl_dir))
        self.e_rel_i.set(e.release_i)
        self.e_rel_j.set(e.release_j)
        self._refresh_pl_list(e)
        self.tabs.select(self.tab_props)

    def _refresh_pl_list(self, e):
        self.pl_list.delete(0, "end")
        for p in e.point_loads:
            d = p[2] if len(p) > 2 else "local"
            self.pl_list.insert("end", f"P={p[0]:g} N @ a={p[1]:g} m  [{dir_label(d)}]")

    # ------------------------------------------------------------------
    # Editors: apply changes
    # ------------------------------------------------------------------
    def _apply_node_props(self):
        n = self.node_editor_target
        top = self.winfo_toplevel()
        if n is None:
            messagebox.showinfo("No node selected", "Select a node first (Select/Edit mode, "
                                                    "or click it in the Nodes list).", parent=top)
            return
        try:
            new_x = float(self.n_x.get())
            new_y = float(self.n_y.get())
            new_name = self.n_name.get().strip()
            if new_name:
                n.name = new_name
            n.support = self.n_support.get()
            # a Fixed support holds rotation whatever the tick box says
            n.restrain_rotation = bool(self.n_rot.get()) or n.support == "Fixed"
            n.Fx = float(self.n_fx.get())
            n.Fy = float(self.n_fy.get())
            n.M = float(self.n_m.get())
        except ValueError:
            messagebox.showerror("Invalid input", "X, Y, Fx, Fy, M must be numeric.", parent=top)
            return

        if new_x != n.x or new_y != n.y:
            connected = [e for e in self.elements if e.ni is n or e.nj is n]
            if connected:
                ok = messagebox.askyesno(
                    "Move node",
                    f"Node {n.label()} is used by {len(connected)} element(s). Moving it "
                    "will change the length and/or angle of those elements. Continue?",
                    parent=top)
                if not ok:
                    self._load_node_editor(n)  # revert displayed fields
                    return
            n.x, n.y = new_x, new_y

        self._invalidate_result()
        self._refresh_lists()
        self._load_node_editor(n)
        self._redraw()

    def _apply_elem_props(self):
        e = self.elem_editor_target
        top = self.winfo_toplevel()
        if e is None:
            messagebox.showinfo("No element selected", "Select an element first.", parent=top)
            return
        try:
            new_ni_x = float(self.e_ni_x.get())
            new_ni_y = float(self.e_ni_y.get())
            new_nj_x = float(self.e_nj_x.get())
            new_nj_y = float(self.e_nj_y.get())
            new_length = float(self.e_length.get())
            new_angle_deg = float(self.e_angle.get())
            new_E = float(self.e_E.get())
            new_A = float(self.e_A.get())
            new_I = float(self.e_I.get())
            new_udl = float(self.e_udl.get())
            new_w1 = float(self.e_w1.get())
            new_w2 = float(self.e_w2.get())
        except ValueError:
            messagebox.showerror("Invalid input", "All fields must be numeric.", parent=top)
            return

        if new_length <= 0:
            messagebox.showerror("Invalid length", "Length must be positive.", parent=top)
            return

        current_length = e.length()
        current_angle_deg = np.degrees(np.arctan2(e.nj.y - e.ni.y, e.nj.x - e.ni.x))

        ni_moved = (abs(new_ni_x - e.ni.x) > 1e-9 or abs(new_ni_y - e.ni.y) > 1e-9)
        length_changed = abs(new_length - current_length) > 1e-9
        angle_changed = abs(((new_angle_deg - current_angle_deg + 180) % 360) - 180) > 1e-9
        # node j X/Y only takes effect if length/angle were left unchanged --
        # length/angle fields take priority, as noted in the UI, since editing
        # both node j X/Y AND length/angle at once would be ambiguous.
        nj_xy_moved = (abs(new_nj_x - e.nj.x) > 1e-9 or abs(new_nj_y - e.nj.y) > 1e-9)
        use_length_angle = length_changed or angle_changed

        geometry_changing = ni_moved or use_length_angle or nj_xy_moved
        if geometry_changing:
            affected_via_ni = [oe for oe in self.elements
                               if oe is not e and (oe.ni is e.ni or oe.nj is e.ni)] if ni_moved else []
            target_j_node = e.nj
            affected_via_nj = [oe for oe in self.elements
                               if oe is not e and (oe.ni is target_j_node or oe.nj is target_j_node)] \
                if (use_length_angle or nj_xy_moved) else []
            shared = affected_via_ni + affected_via_nj
            if shared:
                names = sorted(set(s.label() for s in shared))
                ok = messagebox.askyesno(
                    "Shared node(s)",
                    f"This will also move {len(names)} other element(s) that share a node "
                    f"with this one ({', '.join(names)}). Continue?", parent=top)
                if not ok:
                    self._load_elem_editor(e)  # revert displayed fields
                    return

            if ni_moved:
                e.ni.x, e.ni.y = new_ni_x, new_ni_y

            if use_length_angle:
                rad = np.radians(new_angle_deg)
                e.nj.x = e.ni.x + new_length * np.cos(rad)
                e.nj.y = e.ni.y + new_length * np.sin(rad)
            elif nj_xy_moved:
                e.nj.x, e.nj.y = new_nj_x, new_nj_y
            elif ni_moved:
                # node i moved but node j field wasn't touched and length/angle
                # weren't touched either -> keep node j fixed in place (so the
                # element's length/angle change as a natural consequence).
                pass

        e.E, e.A, e.I = new_E, new_A, new_I
        e.udl, e.w1, e.w2 = new_udl, new_w1, new_w2
        e.udl_dir = dir_tag(self.e_udl_dir.get())
        e.lvl_dir = dir_tag(self.e_lvl_dir.get())
        e.release_i = bool(self.e_rel_i.get())
        e.release_j = bool(self.e_rel_j.get())

        self._invalidate_result()
        self._refresh_lists()
        self._load_elem_editor(e)
        self._redraw()

    def _add_point_load(self):
        e = self.elem_editor_target
        top = self.winfo_toplevel()
        if e is None:
            messagebox.showinfo("No element selected", "Select an element first.", parent=top)
            return
        try:
            P = float(self.pl_P.get())
            a = float(self.pl_a.get())
        except ValueError:
            messagebox.showerror("Invalid input", "P and a must be numeric.", parent=top)
            return
        L = e.length()
        if not (0 <= a <= L):
            messagebox.showerror("Invalid position", f"a must be between 0 and the element "
                                                     f"length ({L:.3f} m).", parent=top)
            return
        e.point_loads.append([P, a, dir_tag(self.pl_dir.get())])
        self._refresh_pl_list(e)
        self._invalidate_result()
        self._redraw()

    def _remove_point_load(self):
        e = self.elem_editor_target
        sel = self.pl_list.curselection()
        if e is None or not sel:
            return
        idx = sel[0]
        del e.point_loads[idx]
        self.pl_list.delete(idx)
        self._invalidate_result()
        self._redraw()

    # ------------------------------------------------------------------
    # Delete / clear / model IO
    # ------------------------------------------------------------------
    def _delete_selected(self):
        if self.selected_node is not None:
            n = self.selected_node
            self.elements = [e for e in self.elements if e.ni is not n and e.nj is not n]
            self.nodes.remove(n)
            self.selected_node = None
            self.node_editor_target = None
        elif self.selected_element is not None:
            self.elements.remove(self.selected_element)
            self.selected_element = None
            self.elem_editor_target = None
        self._invalidate_result()
        self._refresh_lists()
        self._redraw()

    def _clear_all(self, confirm=True):
        if confirm and not messagebox.askyesno("Clear all", "Remove all nodes and elements?",
                                               parent=self.winfo_toplevel()):
            return
        self.nodes.clear()
        self.elements.clear()
        self.selected_node = None
        self.selected_element = None
        self.node_editor_target = None
        self.elem_editor_target = None
        self.result = None
        self.struct = None
        self.fe_elements = {}
        self.show_deformed.set(False)
        self._refresh_lists()
        self._redraw()
        self._set_results_text("Click SOLVE STRUCTURE to see results here.")

    def set_model(self, nodes, elements, fit_view=True):
        """Replace the current model (used by the Beam Chain Builder tab)."""
        self._clear_all(confirm=False)
        self.nodes = list(nodes)
        self.elements = list(elements)
        self._refresh_lists()
        if fit_view:
            self._fit_view_to_structure()
        self._redraw()

    def save_model(self):
        top = self.winfo_toplevel()
        if not self.nodes:
            messagebox.showinfo("Nothing to save", "The model is empty.", parent=top)
            return
        path = filedialog.asksaveasfilename(
            parent=top, defaultextension=".json",
            filetypes=[("Frame model", "*.json"), ("All files", "*.*")],
            title="Save frame model")
        if not path:
            return
        node_index = {n: i for i, n in enumerate(self.nodes)}
        data = {"version": 1,
                "nodes": [n.to_dict() for n in self.nodes],
                "elements": [e.to_dict(node_index) for e in self.elements]}
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError as ex:
            messagebox.showerror("Save failed", str(ex), parent=top)

    def load_model_dialog(self):
        top = self.winfo_toplevel()
        path = filedialog.askopenfilename(
            parent=top, filetypes=[("Frame model", "*.json"), ("All files", "*.*")],
            title="Load frame model")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            nodes = []
            for nd in data["nodes"]:
                n = NodeItem(float(nd["x"]), float(nd["y"]))
                if nd.get("name"):
                    n.name = str(nd["name"])
                dn = nd.get("dof_names") or ["", "", ""]
                n.dof_names = [str(x) for x in (list(dn) + ["", "", ""])[:3]]
                n.support = nd.get("support", "Free")
                # older files have no rotation flag: Fixed held it, nothing else did
                n.restrain_rotation = bool(nd.get("restrain_rotation",
                                                  n.support == "Fixed"))
                n.Fx, n.Fy, n.M = (float(nd.get("Fx", 0)), float(nd.get("Fy", 0)),
                                   float(nd.get("M", 0)))
                nodes.append(n)
            elements = []
            for ed in data["elements"]:
                e = ElementItem(nodes[ed["ni"]], nodes[ed["nj"]])
                e.E, e.A, e.I = float(ed["E"]), float(ed["A"]), float(ed["I"])
                e.udl, e.w1, e.w2 = (float(ed.get("udl", 0)), float(ed.get("w1", 0)),
                                     float(ed.get("w2", 0)))
                # models saved before load directions / releases existed default
                # to the old behaviour: perpendicular loads, rigid connections
                e.udl_dir = ed.get("udl_dir", "local")
                e.lvl_dir = ed.get("lvl_dir", "local")
                e.release_i = bool(ed.get("release_i", False))
                e.release_j = bool(ed.get("release_j", False))
                rn = ed.get("release_names") or ["", ""]
                e.release_names = [str(x) for x in (list(rn) + ["", ""])[:2]]
                e.point_loads = [[float(p[0]), float(p[1]),
                                  p[2] if len(p) > 2 else "local"]
                                 for p in ed.get("point_loads", [])]
                elements.append(e)
        except (OSError, KeyError, ValueError, TypeError, IndexError) as ex:
            messagebox.showerror("Load failed", f"Could not read that model file:\n{ex}",
                                 parent=top)
            return
        self.set_model(nodes, elements)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def _build_fe_model(self):
        """
        Translate the GUI model into fe_engine objects.

        Used both by solve() and by the per-element working tabs, so the
        working shown before a solve is built from exactly the same objects
        the solver would use.
        """
        fe_nodes = []
        node_map = {}
        for n in self.nodes:
            support, axis = split_support(n.support)
            fen = Node(n.id, n.x, n.y, support=support, roller_axis=axis,
                       restrain_rotation=n.restrain_rotation, name=n.name)
            fen.dof_names = list(n.dof_names)
            fen.loads = [n.Fx, n.Fy, n.M]
            fe_nodes.append(fen)
            node_map[n] = fen

        fe_elements = []
        mapping = {}
        for e in self.elements:
            fee = Element(e.id, node_map[e.ni], node_map[e.nj],
                          E=e.E, A=e.A, I=e.I,
                          udl=e.udl, w1=e.w1, w2=e.w2,
                          udl_dir=e.udl_dir, lvl_dir=e.lvl_dir,
                          release_i=e.release_i, release_j=e.release_j,
                          point_loads=[tuple(p) for p in e.point_loads])
            fee.release_names = list(e.release_names)
            fe_elements.append(fee)
            mapping[e] = fee

        return Structure(fe_nodes, fe_elements), mapping

    def _report_context(self):
        """(struct, {item: fe element}, result) for the working tabs."""
        if self.result is not None and self.struct is not None and self.fe_elements:
            return self.struct, self.fe_elements, self.result
        if not self.elements:
            return None, {}, None
        try:
            struct, mapping = self._build_fe_model()
        except Exception:
            return None, {}, None
        return struct, mapping, None

    def solve(self):
        top = self.winfo_toplevel()
        if len(self.nodes) < 2 or len(self.elements) < 1:
            messagebox.showwarning("Not enough data", "Add at least 2 nodes and 1 element.",
                                   parent=top)
            return False

        if not any(n.is_supported() for n in self.nodes):
            messagebox.showwarning("No supports", "Add at least one support before solving.",
                                   parent=top)
            return False

        struct, mapping = self._build_fe_model()
        self.fe_elements = mapping
        try:
            result = struct.solve()
        except np.linalg.LinAlgError as ex:
            messagebox.showerror("Solve failed", str(ex), parent=top)
            return False
        except Exception as ex:
            messagebox.showerror("Solve failed", f"Unexpected error: {ex}", parent=top)
            return False

        self.result = result
        self.struct = struct
        self.show_deformed.set(True)
        self._auto_scale(redraw=False)
        self._display_results()
        self._sync_element_tabs()
        self._redraw()
        self.tabs.select(self.tab_results)
        return True

    def _display_results(self):
        r = self.result
        lines = []
        lines.append("=" * 66)
        lines.append("NODAL DISPLACEMENTS")
        lines.append("=" * 66)
        labels = self.struct.dof_labels(self.dof_style_tag())
        restrained = set(r["restrained"])
        lines.append("  dof  label            value                  ")
        lines.append("  " + "-" * 62)
        for idx, n in enumerate(self.nodes):
            for c, (unit, mult) in enumerate([("mm", 1e3), ("mm", 1e3), ("mrad", 1e3)]):
                d = 3*idx + c
                held = " (restrained)" if d in restrained else ""
                lines.append(f"  {d:>3}  {labels[d]:<16} "
                             f"{r['q'][d, 0]*mult:12.5f} {unit}{held}")
        extra = [d for d in range(3*len(self.nodes), self.struct.ndof)]
        for d in extra:
            lines.append(f"  {d:>3}  {labels[d]:<16} "
                         f"{r['q'][d, 0]*1e3:12.5f} mrad   (hinged member end)")
        lines.append("")
        free = [d for d in range(self.struct.ndof) if d not in restrained]
        lines.append("Free DOFs -- these are the ones that appear in the displacement")
        lines.append("vector that gets solved:")
        lines.append("   " + (", ".join(f"{labels[d]} ({d})" for d in free) or "(none)"))

        lines.append("")
        lines.append("=" * 66)
        lines.append("SUPPORT REACTIONS")
        lines.append("=" * 66)
        any_support = False
        for idx, n in enumerate(self.nodes):
            if not n.is_supported():
                continue
            any_support = True
            Rx, Ry, M = r["R"][[3*idx, 3*idx+1, 3*idx+2], 0]
            tx, ty, rot = n.restraints()
            # only restrained DOFs carry a reaction; blank the others so the
            # table cannot be misread
            sx = f"{Rx:12.2f} N" if tx else "        --  "
            sy = f"{Ry:12.2f} N" if ty else "        --  "
            sm = f"{M:12.2f} Nm" if rot else "        --   "
            lines.append(f"{n.label():>4} [{n.support_text():>16}]  "
                         f"Rx={sx}   Ry={sy}   M={sm}")
        if not any_support:
            lines.append("(none)")
        lines.append("('--' means that DOF is free, so it carries no reaction.")
        lines.append(" A roller with its rotation restrained is a guided support:")
        lines.append(" it slides along its free axis but does carry a moment.)")

        held = sorted(restrained)
        if held:
            lines.append("")
            lines.append("Support reaction vector, one row per restrained DOF:")
            lines.append("  dof  label            reaction")
            lines.append("  " + "-" * 48)
            for dof in held:
                unit = "Nm" if dof % 3 == 2 else "N"
                lines.append(f"  {dof:>3}  {labels[dof]:<16} "
                             f"{r['R'][dof, 0]:14.2f} {unit}")

        lines.append("")
        lines.append("=" * 66)
        lines.append("MEMBER END FORCES (local axes: axial, shear, moment)")
        lines.append("=" * 66)
        for e in self.elements:
            Fl = r["elem_forces"][e.id]["local"].ravel()
            lines.append(f"{e.label()}")
            hi = "  <- hinged, moment released" if e.release_i else ""
            hj = "  <- hinged, moment released" if e.release_j else ""
            lines.append(f"   End i: N={Fl[0]:10.2f} N   V={Fl[1]:10.2f} N   M={Fl[2]:10.2f} Nm{hi}")
            lines.append(f"   End j: N={Fl[3]:10.2f} N   V={Fl[4]:10.2f} N   M={Fl[5]:10.2f} Nm{hj}")

        if any(e.release_i or e.release_j for e in self.elements):
            lines.append("")
            lines.append("=" * 66)
            lines.append("JOINT CONNECTIONS")
            lines.append("=" * 66)
            by_node = {}
            for e in self.elements:
                by_node.setdefault(e.ni, []).append((e, "i", e.release_i))
                by_node.setdefault(e.nj, []).append((e, "j", e.release_j))
            for n in self.nodes:
                members = by_node.get(n, [])
                if len(members) < 2 and not any(rel for _, _, rel in members):
                    continue
                rigid = [f"E{e.id}" for e, _end, rel in members if not rel]
                hinged = [f"E{e.id}" for e, _end, rel in members if rel]
                lines.append(f"{n.label():>4}  rigidly connected: "
                             f"{', '.join(rigid) if rigid else '(none)'}"
                             f"   hinged: {', '.join(hinged) if hinged else '(none)'}")
                if not rigid and len(members) > 0:
                    lines.append("       every member here is hinged, so the joint carries "
                                 "no moment at all")

        lines.append("")
        lines.append("=" * 66)
        lines.append("PEAK MEMBER DEFLECTIONS (relative to the member chord)")
        lines.append("=" * 66)
        for e in self.elements:
            fee = self.fe_elements.get(e)
            if fee is None:
                continue
            X, Y = self.struct.deflected_shape(fee, r["q"], npts=101, scale=1.0)
            # distance of the deflected curve from the straight line joining
            # the two displaced end points
            x0, y0, x1, y1 = X[0], Y[0], X[-1], Y[-1]
            dx, dy = x1 - x0, y1 - y0
            span = np.hypot(dx, dy)
            if span == 0:
                continue
            # measured in the member's local +y sense, so a sagging member
            # reports a negative deflection, matching the load convention
            off = ((Y - y0) * dx - (X - x0) * dy) / span
            k = int(np.argmax(np.abs(off)))
            lines.append(f"{e.label():>18}  max local deflection = {off[k]*1e3:9.4f} mm "
                         f"at {k/(len(off)-1)*e.length():6.3f} m from node i")

        lines.append("")
        lines.append("=" * 66)
        lines.append("EQUILIBRIUM CHECK")
        lines.append("=" * 66)
        sumRx = sum(r["R"][3*i, 0] for i in range(len(self.nodes)))
        sumRy = sum(r["R"][3*i+1, 0] for i in range(len(self.nodes)))
        sumAppliedFx = sum(n.Fx for n in self.nodes)
        sumAppliedFy = sum(n.Fy for n in self.nodes)
        # distributed + point load resultants, resolved to global. Each load
        # carries its own direction, so ask the element for the total.
        distFx = distFy = 0.0
        for e in self.elements:
            fee = self.fe_elements.get(e)
            if fee is None:
                continue
            fx, fy = fee.resultant_global()
            distFx += fx
            distFy += fy
        lines.append(f"Sum of reactions Rx = {sumRx:12.2f} N")
        lines.append(f"Sum of applied Fx (nodal + distributed/point, global) = "
                     f"{sumAppliedFx + distFx:12.2f} N")
        lines.append(f"Sum of reactions Ry = {sumRy:12.2f} N")
        lines.append(f"Sum of applied Fy (nodal + distributed/point, global) = "
                     f"{sumAppliedFy + distFy:12.2f} N")
        lines.append("(Reactions should be equal and opposite to applied loads.)")

        if r.get("warnings"):
            lines.insert(0, "")
            for w in reversed(r["warnings"]):
                lines.insert(0, f"  ! {w}")
            lines.insert(0, "WARNINGS")

        self._set_results_text("\n".join(lines))

    def results_as_text(self):
        return self.results_text.get("1.0", "end-1c")

    def _set_results_text(self, text):
        self._set_text(self.results_text, text)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _redraw(self):
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle=":", alpha=0.5)
        # Fixed, manual view window -- never autoscales to content, so
        # clicking to add a node outside existing geometry always works.
        try:
            xmin, xmax = self.view_xmin.get(), self.view_xmax.get()
            ymin, ymax = self.view_ymin.get(), self.view_ymax.get()
            if xmax > xmin and ymax > ymin:
                self.ax.set_xlim(xmin, xmax)
                self.ax.set_ylim(ymin, ymax)
        except tk.TclError:
            pass  # entry box temporarily empty/invalid while typing

        # elements (undeformed)
        for e in self.elements:
            color, lw = ("tab:red", 3.5) if e is self.selected_element else ("tab:blue", 2.5)
            self.ax.plot([e.ni.x, e.nj.x], [e.ni.y, e.nj.y], color=color, lw=lw, zorder=2)
            mx, my = (e.ni.x + e.nj.x) / 2, (e.ni.y + e.nj.y) / 2
            self.ax.annotate(e.label().split(":")[0], (mx, my), fontsize=8,
                             color="tab:blue", ha="center", va="center",
                             bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7))
            # load indicator glyphs. The uniform and varying loads can point
            # in different directions, so they are drawn separately.
            if e.udl != 0:
                self._draw_load_arrows(e, e.udl, e.udl, e.udl_dir)
            if e.w1 != 0 or e.w2 != 0:
                self._draw_load_arrows(e, e.w1, e.w2, e.lvl_dir)
            for p in e.point_loads:
                self._draw_point_load_arrow(e, p[0], p[1],
                                            p[2] if len(p) > 2 else "local")
            self._draw_hinges(e)

        # deformed shape -- a true curve, so UDL / LVL / point-load bending
        # is visible even when the end nodes barely move
        if self.show_deformed.get() and self.result is not None and self.struct is not None:
            scale = self.deformed_scale.get()
            for e in self.elements:
                fee = self.fe_elements.get(e)
                if fee is None:
                    continue
                X, Y = self.struct.deflected_shape(fee, self.result["q"],
                                                   npts=DEFORMED_NPTS, scale=scale)
                self.ax.plot(X, Y, color="tab:orange", lw=2, ls="-", zorder=3, alpha=0.95)
                self.ax.plot([X[0], X[-1]], [Y[0], Y[-1]], color="tab:orange",
                             lw=0.8, ls=":", zorder=3, alpha=0.5)

        # nodes
        for n in self.nodes:
            color = "tab:red" if n is self.selected_node else "black"
            self.ax.scatter([n.x], [n.y], s=60, color=color, zorder=4)
            self.ax.annotate(n.label(), (n.x, n.y), fontsize=8, xytext=(6, 6),
                             textcoords="offset points")
            self._draw_support_symbol(n)
            if n.Fx != 0 or n.Fy != 0 or n.M != 0:
                self._draw_nodal_load_arrow(n)

        if self.pending_node_for_element is not None:
            n = self.pending_node_for_element
            self.ax.scatter([n.x], [n.y], s=140, facecolors="none",
                            edgecolors="tab:green", linewidths=2, zorder=5)

        self.ax.set_title("Click canvas to add nodes / elements  -  "
                          "orange = deformed shape" if self.show_deformed.get()
                          else "Click canvas to add nodes / elements")
        self.canvas.draw_idle()

    def _draw_support_symbol(self, n):
        """
        Drawn from the restraints actually applied, not from the type name, so
        the glyph never disagrees with the maths. A roller whose rotation is
        held gets the guided-support symbol: a rigid bar on rollers, rather
        than a triangle pinned at the node.
        """
        s = 0.3
        tx, ty, rot = n.restraints()

        if tx and ty and rot:                       # fully fixed
            self.ax.plot([n.x - s, n.x + s], [n.y - s, n.y - s], color="black", lw=2)
            for i in range(5):
                xx = n.x - s + i * (2*s/4)
                self.ax.plot([xx, xx - 0.08], [n.y - s, n.y - s - 0.15], color="black", lw=1)
        elif tx and ty:                             # pinned
            self.ax.plot([n.x, n.x - s*0.6, n.x + s*0.6, n.x],
                         [n.y, n.y - s, n.y - s, n.y], color="black", lw=1.5)
        elif ty and not rot:                        # roller, rolls horizontally
            self.ax.plot([n.x, n.x - s*0.6, n.x + s*0.6, n.x],
                         [n.y, n.y - s, n.y - s, n.y], color="black", lw=1.5)
            self.ax.plot([n.x - s*0.6, n.x + s*0.6], [n.y - s - 0.08, n.y - s - 0.08],
                         color="black", lw=1.5)
            for dx in (-s*0.3, s*0.3):
                self.ax.add_patch(plt_circle(n.x + dx, n.y - s - 0.04, 0.04))
        elif ty and rot:                            # guided: slides in X, no rotation
            self.ax.plot([n.x - s*0.7, n.x + s*0.7], [n.y, n.y], color="black", lw=2.5)
            self.ax.plot([n.x, n.x], [n.y, n.y - s*0.35], color="black", lw=1.5)
            self.ax.plot([n.x - s*0.7, n.x + s*0.7],
                         [n.y - s*0.55, n.y - s*0.55], color="black", lw=1.5)
            for dx in (-s*0.35, s*0.35):
                self.ax.add_patch(plt_circle(n.x + dx, n.y - s*0.45, 0.045))
        elif tx and not rot:                        # roller, rolls vertically
            self.ax.plot([n.x, n.x - s, n.x - s, n.x],
                         [n.y, n.y - s*0.6, n.y + s*0.6, n.y], color="black", lw=1.5)
            self.ax.plot([n.x - s - 0.08, n.x - s - 0.08], [n.y - s*0.6, n.y + s*0.6],
                         color="black", lw=1.5)
            for dy in (-s*0.3, s*0.3):
                self.ax.add_patch(plt_circle(n.x - s - 0.04, n.y + dy, 0.04))
        elif tx and rot:                            # guided: slides in Y, no rotation
            self.ax.plot([n.x, n.x], [n.y - s*0.7, n.y + s*0.7], color="black", lw=2.5)
            self.ax.plot([n.x, n.x - s*0.35], [n.y, n.y], color="black", lw=1.5)
            self.ax.plot([n.x - s*0.55, n.x - s*0.55],
                         [n.y - s*0.7, n.y + s*0.7], color="black", lw=1.5)
            for dy in (-s*0.35, s*0.35):
                self.ax.add_patch(plt_circle(n.x - s*0.45, n.y + dy, 0.045))
        elif rot:                                   # rotation only (symmetry line)
            r = s * 0.45
            self.ax.plot([n.x - r, n.x + r, n.x + r, n.x - r, n.x - r],
                         [n.y - r, n.y - r, n.y + r, n.y + r, n.y - r],
                         color="black", lw=1.5)

    def _draw_nodal_load_arrow(self, n):
        scale = 0.6
        if n.Fx != 0:
            self.ax.annotate("", xy=(n.x + scale * np.sign(n.Fx), n.y), xytext=(n.x, n.y),
                             arrowprops=dict(arrowstyle="->", color="tab:purple", lw=2))
        if n.Fy != 0:
            self.ax.annotate("", xy=(n.x, n.y + scale * np.sign(n.Fy)), xytext=(n.x, n.y),
                             arrowprops=dict(arrowstyle="->", color="tab:purple", lw=2))
        if n.M != 0:
            self.ax.annotate("M", (n.x + 0.15, n.y + 0.15), color="tab:purple",
                             fontsize=9, fontweight="bold")

    @staticmethod
    def _load_dir_vector(e, direction):
        """Global unit vector a POSITIVE load of this kind acts along."""
        alpha = np.arctan2(e.nj.y - e.ni.y, e.nj.x - e.ni.x)
        if direction == "gx":
            return (1.0, 0.0)
        if direction == "gy":
            return (0.0, 1.0)
        return (-np.sin(alpha), np.cos(alpha))   # local +y

    def _draw_load_arrows(self, e, w_i, w_j, direction, n_arrows=6):
        """Arrows for one distributed load, drawn along its actual direction."""
        w_max = max(abs(w_i), abs(w_j))
        if w_max == 0:
            return
        dx, dy = self._load_dir_vector(e, direction)
        tips = []
        for k in range(n_arrows):
            t = k / (n_arrows - 1) if n_arrows > 1 else 0.5
            w = w_i * (1 - t) + w_j * t
            x0 = e.ni.x + t * (e.nj.x - e.ni.x)
            y0 = e.ni.y + t * (e.nj.y - e.ni.y)
            # Arrow length proportional to intensity, so an LVL reads as a
            # wedge. The tail sits on the side the load comes from and the
            # head touches the member, so a negative (downward) load draws
            # as arrows pressing down onto it.
            mag = -0.45 * w / w_max
            x1, y1 = x0 + mag * dx, y0 + mag * dy
            tips.append((x1, y1))
            if w != 0:
                self.ax.annotate("", xy=(x0, y0), xytext=(x1, y1),
                                 arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.3))
        # outline joining the arrow tails shows the load profile shape
        self.ax.plot([p[0] for p in tips], [p[1] for p in tips],
                     color="tab:green", lw=1.0, alpha=0.8)

    def _draw_point_load_arrow(self, e, P, a, direction="local"):
        if P == 0:
            return
        L = e.length()
        t = a / L if L > 0 else 0
        dx, dy = self._load_dir_vector(e, direction)
        x0 = e.ni.x + t * (e.nj.x - e.ni.x)
        y0 = e.ni.y + t * (e.nj.y - e.ni.y)
        mag = -0.55 * np.sign(P)
        self.ax.annotate("", xy=(x0, y0), xytext=(x0 + mag * dx, y0 + mag * dy),
                         arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.2))

    def _draw_hinges(self, e):
        """Open circle just inside a released member end, the usual hinge symbol."""
        L = e.length()
        if L <= 0:
            return
        ux, uy = (e.nj.x - e.ni.x) / L, (e.nj.y - e.ni.y) / L
        off = min(0.18, 0.15 * L)
        r = min(0.07, 0.06 * L)
        if e.release_i:
            self.ax.add_patch(plt_circle(e.ni.x + off * ux, e.ni.y + off * uy, r,
                                         face="white"))
        if e.release_j:
            self.ax.add_patch(plt_circle(e.nj.x - off * ux, e.nj.y - off * uy, r,
                                         face="white"))


def plt_circle(x, y, r, face=None):
    from matplotlib.patches import Circle
    if face is None:
        return Circle((x, y), r, fill=False, color="black", lw=1.0, zorder=5)
    return Circle((x, y), r, facecolor=face, edgecolor="black", lw=1.4, zorder=5)


def main():
    root = tk.Tk()
    root.title("2D Frame Designer & Solver")
    root.geometry("1400x900")
    panel = FrameDesignerPanel(root)
    panel.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
