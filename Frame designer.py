"""
2D Frame Analysis Tool — interactive GUI
=========================================

Click on the canvas to place nodes, click two nodes to connect them into a
beam/column element, assign section properties, supports (Fixed / Pinned /
Roller), nodal point loads, UDLs, point loads, and linearly varying
(trapezoidal) loads on each element, then solve for displacements,
reactions and member end forces, and view the deformed shape.

Run with:  python frame_designer.py
Requires:  numpy, matplotlib   (tkinter ships with standard Python on
           Windows/Mac; on Linux install via your package manager, e.g.
           `sudo apt install python3-tk` if it's missing)
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from fe_engine import Node, Element, Structure

SUPPORT_TYPES = ["Free", "Pinned", "Roller", "Fixed"]
GRID_SNAP = 0.25  # metres, snap tolerance when clicking
DEFAULT_VIEW_HALF_RANGE = 10.0  # metres; fixed canvas view, does not autoscale


# ----------------------------------------------------------------------
# Data model wrapper (keeps GUI-friendly extra fields alongside fe_engine)
# ----------------------------------------------------------------------
class NodeItem:
    _next_id = 0

    def __init__(self, x, y):
        self.id = NodeItem._next_id
        NodeItem._next_id += 1
        self.x = x
        self.y = y
        self.support = "Free"
        self.Fx = 0.0
        self.Fy = 0.0
        self.M = 0.0

    def label(self):
        return f"N{self.id}"


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
        self.udl = 0.0          # N/m, uniform, downward negative (local -y)
        self.w1 = 0.0           # trapezoidal load intensity at node i
        self.w2 = 0.0           # trapezoidal load intensity at node j
        self.point_loads = []   # list of [P, a]  (P in N, a in m from node i)

    def label(self):
        return f"E{self.id}: N{self.ni.id}-N{self.nj.id}"

    def length(self):
        return np.hypot(self.nj.x - self.ni.x, self.nj.y - self.ni.y)


# ----------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------
class FrameDesignerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Frame Designer & Solver")
        self.root.geometry("1400x900")

        self.nodes = []
        self.elements = []
        self.mode = tk.StringVar(value="node")   # node | element | select
        self.pending_node_for_element = None
        self.selected_node = None
        self.selected_element = None

        self.result = None  # last solve() result dict
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
        main = ttk.Frame(self.root)
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

        ttk.Checkbutton(toolbar, text="Show deformed shape",
                         variable=self.show_deformed,
                         command=self._redraw).pack(side="right", padx=4)
        ttk.Label(toolbar, text="Scale:").pack(side="right")
        ttk.Spinbox(toolbar, from_=1, to=5000, increment=10, width=6,
                    textvariable=self.deformed_scale,
                    command=self._redraw).pack(side="right", padx=4)

        self.fig = Figure(figsize=(8, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        solve_bar = ttk.Frame(left)
        solve_bar.pack(side="bottom", fill="x", padx=4, pady=6)
        ttk.Button(solve_bar, text="SOLVE STRUCTURE", command=self._solve,
                   style="Solve.TButton").pack(side="left", fill="x", expand=True)
        style = ttk.Style()
        style.configure("Solve.TButton", font=("TkDefaultFont", 11, "bold"))

        # Right: property / results panel
        right = ttk.Frame(main, width=430)
        right.pack(side="right", fill="both")
        right.pack_propagate(False)

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)

        self.tab_props = ttk.Frame(self.tabs)
        self.tab_results = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_props, text="Properties")
        self.tabs.add(self.tab_results, text="Results")

        self._build_props_tab()
        self._build_results_tab()

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
        # this panel, so it doesn't hijack scrolling elsewhere (e.g. the
        # results text box or listboxes have their own scroll handling).
        scroll_canvas.bind("<Enter>", _bind_wheel)
        scroll_canvas.bind("<Leave>", _unbind_wheel)

        lst_frame = ttk.LabelFrame(frame, text="Nodes")
        lst_frame.pack(fill="both", padx=6, pady=4)
        self.node_list = tk.Listbox(lst_frame, height=6)
        self.node_list.pack(fill="x", padx=4, pady=4)
        self.node_list.bind("<<ListboxSelect>>", self._on_node_list_select)

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

        ttk.Label(parent, text="UDL (N/m, - = down):").grid(row=row, column=0, sticky="w", padx=4)
        self.e_udl = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.e_udl, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1

        ttk.Label(parent, text="Trapezoid w1 @ node i (N/m):").grid(row=row, column=0, sticky="w", padx=4)
        self.e_w1 = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.e_w1, width=14).grid(row=row, column=1, padx=4, pady=2)
        row += 1

        ttk.Label(parent, text="Trapezoid w2 @ node j (N/m):").grid(row=row, column=0, sticky="w", padx=4)
        self.e_w2 = tk.StringVar(value="0")
        ttk.Entry(parent, textvariable=self.e_w2, width=14).grid(row=row, column=1, padx=4, pady=2)
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
        ttk.Button(pl_add_frame, text="Add", command=self._add_point_load).pack(side="left", padx=4)
        ttk.Button(pl_add_frame, text="Remove Sel.", command=self._remove_point_load).pack(side="left")
        row += 1

        ttk.Button(parent, text="Apply to Selected Element",
                   command=self._apply_elem_props).grid(row=row, column=0, columnspan=2,
                                                          sticky="ew", padx=4, pady=6)

    def _build_results_tab(self):
        frame = self.tab_results
        self.results_text = tk.Text(frame, wrap="word", font=("Courier New", 9))
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=vsb.set)
        self.results_text.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        vsb.pack(side="right", fill="y", pady=6)
        self.results_text.insert("1.0", "Click SOLVE STRUCTURE to see results here.")
        self.results_text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Canvas interaction
    # ------------------------------------------------------------------
    def _reset_pending(self):
        self.pending_node_for_element = None
        self._redraw()

    def _add_node_by_coords_dialog(self):
        x = simpledialog.askfloat("New Node", "X coordinate (m):", parent=self.root)
        if x is None:
            return
        y = simpledialog.askfloat("New Node", "Y coordinate (m):", parent=self.root)
        if y is None:
            return
        n = NodeItem(x, y)
        self.nodes.append(n)
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
            x1, y1, x2, y2 = e.ni.x, e.ni.y, e.nj.x, e.nj.y
            d = self._point_seg_dist(x, y, x1, y1, x2, y2)
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
                n = NodeItem(sx, sy)
                self.nodes.append(n)
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
                    el = ElementItem(self.pending_node_for_element, n)
                    self.elements.append(el)
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
        for n in self.nodes:
            self.node_list.insert(
                "end",
                f"{n.label()}  ({n.x:.2f}, {n.y:.2f})  [{n.support}]"
            )
        self.elem_list.delete(0, "end")
        for e in self.elements:
            self.elem_list.insert("end", f"{e.label()}  L={e.length():.2f}m")

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

    def _load_node_editor(self, n):
        self.node_editor_target = n
        self.n_coord_lbl.config(text=f"Node: {n.label()}  ({n.x:.2f}, {n.y:.2f})")
        self.n_x.set(f"{n.x:.4f}")
        self.n_y.set(f"{n.y:.4f}")
        self.n_support.set(n.support)
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
        self.pl_list.delete(0, "end")
        for (P, a) in e.point_loads:
            self.pl_list.insert("end", f"P={P} N @ a={a} m")
        self.tabs.select(self.tab_props)

    # ------------------------------------------------------------------
    # Editors: apply changes
    # ------------------------------------------------------------------
    def _apply_node_props(self):
        n = self.node_editor_target
        if n is None:
            messagebox.showinfo("No node selected", "Select a node first (Select/Edit mode, "
                                                      "or click it in the Nodes list).")
            return
        try:
            new_x = float(self.n_x.get())
            new_y = float(self.n_y.get())
            n.support = self.n_support.get()
            n.Fx = float(self.n_fx.get())
            n.Fy = float(self.n_fy.get())
            n.M = float(self.n_m.get())
        except ValueError:
            messagebox.showerror("Invalid input", "X, Y, Fx, Fy, M must be numeric.")
            return

        if (new_x != n.x or new_y != n.y):
            connected = [e for e in self.elements if e.ni is n or e.nj is n]
            if connected:
                ok = messagebox.askyesno(
                    "Move node",
                    f"Node {n.label()} is used by {len(connected)} element(s). Moving it "
                    "will change the length and/or angle of those elements. Continue?"
                )
                if not ok:
                    self._load_node_editor(n)  # revert displayed fields
                    return
            n.x, n.y = new_x, new_y

        self._refresh_lists()
        self._load_node_editor(n)
        self._redraw()

    def _apply_elem_props(self):
        e = self.elem_editor_target
        if e is None:
            messagebox.showinfo("No element selected", "Select an element first.")
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
            messagebox.showerror("Invalid input", "All fields must be numeric.")
            return

        if new_length <= 0:
            messagebox.showerror("Invalid length", "Length must be positive.")
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
                    f"with this one ({', '.join(names)}). Continue?"
                )
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

        self._refresh_lists()
        self._load_elem_editor(e)
        self._redraw()

    def _add_point_load(self):
        e = self.elem_editor_target
        if e is None:
            messagebox.showinfo("No element selected", "Select an element first.")
            return
        try:
            P = float(self.pl_P.get())
            a = float(self.pl_a.get())
        except ValueError:
            messagebox.showerror("Invalid input", "P and a must be numeric.")
            return
        L = e.length()
        if not (0 <= a <= L):
            messagebox.showerror("Invalid position", f"a must be between 0 and the element "
                                                       f"length ({L:.3f} m).")
            return
        e.point_loads.append([P, a])
        self.pl_list.insert("end", f"P={P} N @ a={a} m")

    def _remove_point_load(self):
        e = self.elem_editor_target
        sel = self.pl_list.curselection()
        if e is None or not sel:
            return
        idx = sel[0]
        del e.point_loads[idx]
        self.pl_list.delete(idx)

    # ------------------------------------------------------------------
    # Delete / clear
    # ------------------------------------------------------------------
    def _delete_selected(self):
        if self.selected_node is not None:
            n = self.selected_node
            self.elements = [e for e in self.elements if e.ni is not n and e.nj is not n]
            self.nodes.remove(n)
            self.selected_node = None
        elif self.selected_element is not None:
            self.elements.remove(self.selected_element)
            self.selected_element = None
        self._refresh_lists()
        self._redraw()

    def _clear_all(self):
        if not messagebox.askyesno("Clear all", "Remove all nodes and elements?"):
            return
        self.nodes.clear()
        self.elements.clear()
        self.selected_node = None
        self.selected_element = None
        self.result = None
        self._refresh_lists()
        self._redraw()
        self._set_results_text("Click SOLVE STRUCTURE to see results here.")

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def _solve(self):
        if len(self.nodes) < 2 or len(self.elements) < 1:
            messagebox.showwarning("Not enough data", "Add at least 2 nodes and 1 element.")
            return

        support_count = sum(1 for n in self.nodes if n.support != "Free")
        if support_count == 0:
            messagebox.showwarning("No supports", "Add at least one support before solving.")
            return

        fe_nodes = []
        node_map = {}
        for n in self.nodes:
            fen = Node(n.id, n.x, n.y, support=n.support)
            fen.loads = [n.Fx, n.Fy, n.M]
            fe_nodes.append(fen)
            node_map[n] = fen

        fe_elements = []
        for e in self.elements:
            fee = Element(e.id, node_map[e.ni], node_map[e.nj],
                          E=e.E, A=e.A, I=e.I,
                          udl=e.udl, w1=e.w1, w2=e.w2,
                          point_loads=[(P, a) for (P, a) in e.point_loads])
            fe_elements.append(fee)

        struct = Structure(fe_nodes, fe_elements)
        try:
            result = struct.solve()
        except np.linalg.LinAlgError as ex:
            messagebox.showerror("Solve failed", str(ex))
            return
        except Exception as ex:
            messagebox.showerror("Solve failed", f"Unexpected error: {ex}")
            return

        self.result = result
        self.struct = struct
        self.show_deformed.set(True)
        self._display_results()
        self._redraw()
        self.tabs.select(self.tab_results)

    def _display_results(self):
        r = self.result
        struct = self.struct
        lines = []
        lines.append("=" * 60)
        lines.append("NODAL DISPLACEMENTS")
        lines.append("=" * 60)
        for idx, n in enumerate(self.nodes):
            dofs = [3*idx, 3*idx+1, 3*idx+2]
            u, v, th = r["q"][dofs, 0]
            lines.append(f"{n.label():>4}  ux={u*1e3:10.4f} mm   "
                         f"uy={v*1e3:10.4f} mm   rot={th*1e3:10.5f} mrad")

        lines.append("")
        lines.append("=" * 60)
        lines.append("SUPPORT REACTIONS")
        lines.append("=" * 60)
        for idx, n in enumerate(self.nodes):
            if n.support == "Free":
                continue
            dofs = [3*idx, 3*idx+1, 3*idx+2]
            Rx, Ry, M = r["R"][dofs, 0]
            lines.append(f"{n.label():>4} [{n.support:>7}]  "
                         f"Rx={Rx:12.2f} N   Ry={Ry:12.2f} N   M={M:12.2f} Nm")

        lines.append("")
        lines.append("=" * 60)
        lines.append("MEMBER END FORCES (local axes: axial, shear, moment)")
        lines.append("=" * 60)
        for e in self.elements:
            Fl = r["elem_forces"][e.id]["local"].ravel()
            lines.append(f"{e.label()}")
            lines.append(f"   End i: N={Fl[0]:10.2f} N   V={Fl[1]:10.2f} N   M={Fl[2]:10.2f} Nm")
            lines.append(f"   End j: N={Fl[3]:10.2f} N   V={Fl[4]:10.2f} N   M={Fl[5]:10.2f} Nm")

        lines.append("")
        lines.append("=" * 60)
        lines.append("EQUILIBRIUM CHECK")
        lines.append("=" * 60)
        sumRx = sum(r["R"][3*i, 0] for i in range(len(self.nodes)))
        sumRy = sum(r["R"][3*i+1, 0] for i in range(len(self.nodes)))
        sumAppliedFx = sum(n.Fx for n in self.nodes)
        sumAppliedFy = sum(n.Fy for n in self.nodes)
        # distributed + point load resultants, resolved to global
        distFx = distFy = 0.0
        for e in self.elements:
            L = e.length()
            fy_local = e.udl*L + (e.w1+e.w2)*L/2 + sum(P for P, a in e.point_loads)
            alpha = np.arctan2(e.nj.y - e.ni.y, e.nj.x - e.ni.x)
            c, s = np.cos(alpha), np.sin(alpha)
            distFx += s * fy_local
            distFy += c * fy_local
        lines.append(f"Sum of reactions Rx = {sumRx:12.2f} N")
        lines.append(f"Sum of applied Fx (nodal + distributed/point, global) = "
                     f"{sumAppliedFx + distFx:12.2f} N")
        lines.append(f"Sum of reactions Ry = {sumRy:12.2f} N")
        lines.append(f"Sum of applied Fy (nodal + distributed/point, global) = "
                     f"{sumAppliedFy + distFy:12.2f} N")
        lines.append("(Reactions should be equal and opposite to applied loads.)")

        self._set_results_text("\n".join(lines))

    def _set_results_text(self, text):
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", text)
        self.results_text.configure(state="disabled")

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
            color = "tab:blue"
            lw = 2.5
            if e is self.selected_element:
                color, lw = "tab:red", 3.5
            self.ax.plot([e.ni.x, e.nj.x], [e.ni.y, e.nj.y], color=color, lw=lw, zorder=2)
            mx, my = (e.ni.x + e.nj.x) / 2, (e.ni.y + e.nj.y) / 2
            self.ax.annotate(e.label().split(":")[0], (mx, my), fontsize=8,
                             color="tab:blue", ha="center", va="center",
                             bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7))
            # load indicator glyphs
            if e.udl != 0 or e.w1 != 0 or e.w2 != 0:
                self._draw_load_arrows(e)
            for (P, a) in e.point_loads:
                self._draw_point_load_arrow(e, P, a)

        # deformed shape
        if self.show_deformed.get() and self.result is not None:
            scale = self.deformed_scale.get()
            for e in self.elements:
                idx_i = self.nodes.index(next(n for n in self.nodes if n.id == e.ni.id))
                idx_j = self.nodes.index(next(n for n in self.nodes if n.id == e.nj.id))
                ui = self.result["q"][3*idx_i:3*idx_i+2, 0]
                uj = self.result["q"][3*idx_j:3*idx_j+2, 0]
                xi = e.ni.x + scale * ui[0]
                yi = e.ni.y + scale * ui[1]
                xj = e.nj.x + scale * uj[0]
                yj = e.nj.y + scale * uj[1]
                self.ax.plot([xi, xj], [yi, yj], color="tab:orange", lw=2, ls="--",
                             zorder=3, alpha=0.9)

        # nodes
        for n in self.nodes:
            color = "black"
            if n is self.selected_node:
                color = "tab:red"
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

        self.ax.set_title("Click canvas to add nodes / elements  •  "
                          "orange dashed = deformed shape" if self.show_deformed.get()
                          else "Click canvas to add nodes / elements")
        self.canvas.draw_idle()

    def _draw_support_symbol(self, n):
        s = 0.3
        if n.support == "Fixed":
            self.ax.plot([n.x - s, n.x + s], [n.y - s, n.y - s], color="black", lw=2)
            for i in range(5):
                xx = n.x - s + i * (2*s/4)
                self.ax.plot([xx, xx - 0.08], [n.y - s, n.y - s - 0.15], color="black", lw=1)
        elif n.support == "Pinned":
            tri_x = [n.x, n.x - s*0.6, n.x + s*0.6, n.x]
            tri_y = [n.y, n.y - s, n.y - s, n.y]
            self.ax.plot(tri_x, tri_y, color="black", lw=1.5)
        elif n.support == "Roller":
            tri_x = [n.x, n.x - s*0.6, n.x + s*0.6, n.x]
            tri_y = [n.y, n.y - s, n.y - s, n.y]
            self.ax.plot(tri_x, tri_y, color="black", lw=1.5)
            self.ax.plot([n.x - s*0.6, n.x + s*0.6], [n.y - s - 0.08, n.y - s - 0.08],
                        color="black", lw=1.5)

    def _draw_nodal_load_arrow(self, n):
        scale = 0.6
        if n.Fx != 0:
            dx = scale * np.sign(n.Fx)
            self.ax.annotate("", xy=(n.x + dx, n.y), xytext=(n.x, n.y),
                             arrowprops=dict(arrowstyle="->", color="tab:purple", lw=2))
        if n.Fy != 0:
            dy = scale * np.sign(n.Fy)
            self.ax.annotate("", xy=(n.x, n.y + dy), xytext=(n.x, n.y),
                             arrowprops=dict(arrowstyle="->", color="tab:purple", lw=2))
        if n.M != 0:
            self.ax.annotate("M", (n.x + 0.15, n.y + 0.15), color="tab:purple",
                             fontsize=9, fontweight="bold")

    def _draw_load_arrows(self, e, n_arrows=5):
        L = e.length()
        alpha = np.arctan2(e.nj.y - e.ni.y, e.nj.x - e.ni.x)
        # local -y (perp to member) unit vector, rotated into global
        perp = (np.sin(alpha), -np.cos(alpha))
        for k in range(n_arrows):
            t = k / (n_arrows - 1) if n_arrows > 1 else 0.5
            w = (e.udl + e.w1) * (1 - t) + (e.udl + e.w2) * t
            if w == 0:
                continue
            x0 = e.ni.x + t * (e.nj.x - e.ni.x)
            y0 = e.ni.y + t * (e.nj.y - e.ni.y)
            mag = 0.4 * np.sign(w)
            x1 = x0 - mag * perp[0]
            y1 = y0 - mag * perp[1]
            self.ax.annotate("", xy=(x0, y0), xytext=(x1, y1),
                             arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.3))

    def _draw_point_load_arrow(self, e, P, a):
        if P == 0:
            return
        L = e.length()
        t = a / L if L > 0 else 0
        alpha = np.arctan2(e.nj.y - e.ni.y, e.nj.x - e.ni.x)
        perp = (np.sin(alpha), -np.cos(alpha))
        x0 = e.ni.x + t * (e.nj.x - e.ni.x)
        y0 = e.ni.y + t * (e.nj.y - e.ni.y)
        mag = 0.55 * np.sign(P)
        x1 = x0 - mag * perp[0]
        y1 = y0 - mag * perp[1]
        self.ax.annotate("", xy=(x0, y0), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.2))


def main():
    root = tk.Tk()
    app = FrameDesignerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()