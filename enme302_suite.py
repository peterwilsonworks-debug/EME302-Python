"""
EME302 Structural Analysis Suite
================================

One window that gets at everything in this folder:

  * Frame Designer   -- the interactive click-to-build 2D frame solver
                        (nodes, elements, supports incl. roller axis, UDL /
                        LVL / point loads, reactions, curved deformed shape).
  * Beam Chain Builder -- type in a chain of N beam elements (L, angle, E, A,
                        I, UDL, LVL, point load) plus node supports and nodal
                        loads, then push it straight into the designer and
                        solve. Presets reproduce the lab worksheets.
  * Script Runner    -- run any of the original lab / element scripts in this
                        folder and read their printed matrices, without
                        leaving the GUI.

Run with:  python eme302_suite.py
Requires:  numpy, matplotlib, tkinter
"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np

from frame_designer import (FrameDesignerPanel, NodeItem, ElementItem,
                            SUPPORT_TYPES, DIR_LABELS, dir_label, dir_tag)

PERP = dir_label("local")   # "Perp. to member"
VERT = dir_label("gy")      # "Global Y (vert.)"
HORIZ = dir_label("gx")     # "Global X (horiz.)"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Modules that make up the GUI itself -- not lab scripts to be run
GUI_MODULES = {"eme302_suite.py", "frame_designer.py", "fe_engine.py",
               "Frame designer.py"}


# ======================================================================
# Beam Chain Builder
# ======================================================================
class ChainBuilderPanel(ttk.Frame):
    """
    Parameter-entry front end for a chain of beam elements: element 1 runs
    from node 0 to node 1, element 2 from node 1 to node 2, and so on. This
    is the shape of every worksheet script in this folder (Lab 3, the
    3-element and 4-element models), so those become presets.
    """

    # (key, column header, default, widget kind)
    ELEM_COLS = [
        ("L", "L (m)", "3", "entry"),
        ("angle", "angle (deg)", "0", "entry"),
        ("E", "E (Pa)", "200e9", "entry"),
        ("A", "A (m^2)", "5e-4", "entry"),
        ("I", "I (m^4)", "1e-5", "entry"),
        ("UDL", "UDL (N/m)", "0", "entry"),
        ("UDL_dir", "UDL acts", PERP, "dir"),
        ("w1", "w1 (N/m)", "0", "entry"),
        ("w2", "w2 (N/m)", "0", "entry"),
        ("LVL_dir", "w1/w2 acts", PERP, "dir"),
        ("P", "P (N)", "0", "entry"),
        ("a", "a (m)", "0", "entry"),
        ("P_dir", "P acts", PERP, "dir"),
        ("rel_i", "hinge i", "0", "check"),
        ("rel_j", "hinge j", "0", "check"),
    ]
    NODE_COLS = [("Fx (N)", "0"), ("Fy (N)", "0"), ("M (Nm)", "0")]

    # Presets give only the columns that differ from the defaults above.
    PRESETS = {
        "Lab 3 - 3-beam portal (UDL + centre PL)": {
            "n": 3,
            "elems": [
                {"L": "3", "angle": "90"},
                {"L": "4.5", "angle": "0", "UDL": "-10000", "P": "-50000", "a": "2.25"},
                {"L": "3", "angle": "-90"},
            ],
            "names": ["A", "B", "C", "D"],
            "supports": ["Fixed", "Free", "Free", "Fixed"],
            "nodal": [["0", "0", "0"], ["10000", "0", "0"],
                      ["10000", "0", "0"], ["0", "0", "0"]],
        },
        "3-element portal + LVL on column 3": {
            "n": 3,
            "elems": [
                {"L": "3", "angle": "90"},
                {"L": "4.5", "angle": "0", "UDL": "-10000", "P": "-50000", "a": "2.25"},
                {"L": "3", "angle": "-90", "w2": "-8000"},
            ],
            "names": ["A", "B", "C", "D"],
            "supports": ["Fixed", "Free", "Free", "Fixed"],
            "nodal": [["0", "0", "0"], ["10000", "0", "0"],
                      ["10000", "0", "0"], ["0", "0", "0"]],
        },
        "4-element portal (two UDL spans)": {
            "n": 4,
            "elems": [
                {"L": "3", "angle": "90"},
                {"L": "3", "angle": "0", "UDL": "-10000", "P": "-50000", "a": "1.5"},
                {"L": "3", "angle": "0", "UDL": "-10000"},
                {"L": "3", "angle": "-90"},
            ],
            "names": ["A", "B", "C", "D", "E"],
            "supports": ["Fixed", "Free", "Free", "Free", "Fixed"],
            "nodal": [["0", "0", "0"], ["10000", "0", "0"], ["0", "0", "0"],
                      ["10000", "0", "0"], ["0", "0", "0"]],
        },
        "Pitched portal: VERTICAL UDL on rafters + HINGED apex": {
            "n": 4,
            "elems": [
                {"L": "4", "angle": "90"},
                {"L": "5", "angle": "20", "UDL": "-10000", "UDL_dir": VERT},
                {"L": "5", "angle": "-20", "UDL": "-10000", "UDL_dir": VERT,
                 "rel_i": "1"},
                {"L": "4", "angle": "-90"},
            ],
            "names": ["A", "B", "apex", "D", "E"],
            "supports": ["Fixed", "Free", "Free", "Free", "Fixed"],
            "nodal": [["0", "0", "0"]] * 5,
        },
        "Gerber beam: fixed - internal hinge - roller": {
            "n": 2,
            "elems": [
                {"L": "4", "angle": "0", "UDL": "-10000"},
                {"L": "4", "angle": "0", "UDL": "-10000", "rel_i": "1"},
            ],
            "names": ["A", "hinge", "C"],
            "supports": ["Fixed", "Free", "Roller-Y"],
            "nodal": [["0", "0", "0"]] * 3,
        },
        "Simply supported beam, UDL (pin + roller-Y)": {
            "n": 1,
            "elems": [{"L": "6", "angle": "0", "UDL": "-10000"}],
            "supports": ["Pinned", "Roller-Y"],
            "nodal": [["0", "0", "0"], ["0", "0", "0"]],
        },
        "Propped cantilever, triangular LVL": {
            "n": 1,
            "elems": [{"L": "6", "angle": "0", "w2": "-12000"}],
            "supports": ["Fixed", "Roller-Y"],
            "nodal": [["0", "0", "0"], ["0", "0", "0"]],
        },
        "Vertical mast, roller-X at top": {
            "n": 1,
            "elems": [{"L": "4", "angle": "90", "UDL": "-5000"}],
            "supports": ["Fixed", "Roller-X"],
            "nodal": [["0", "0", "0"], ["0", "0", "0"]],
        },
        "Fixed - GUIDED prop (slides vertically, cannot rotate)": {
            "n": 1,
            "elems": [{"L": "4", "angle": "0", "UDL": "-10000"}],
            "supports": ["Fixed", "Roller-X"],
            "norot": ["0", "1"],       # <- the guided end
            "nodal": [["0", "0", "0"], ["0", "0", "0"]],
        },
    }

    def __init__(self, parent, designer, notebook):
        super().__init__(parent)
        self.designer = designer
        self.notebook = notebook
        self.elem_vars = []   # list of list[StringVar]
        self.node_vars = []   # list of dicts: name/sup/rot/loads vars

        self.n_elements = tk.IntVar(value=3)
        self.start_x = tk.StringVar(value="0")
        self.start_y = tk.StringVar(value="0")
        self.preset = tk.StringVar(value="")

        self._build_layout()
        self._rebuild_rows()

    # ------------------------------------------------------------------
    def _build_layout(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=(8, 4))

        ttk.Label(top, text="Number of elements:").pack(side="left")
        ttk.Spinbox(top, from_=1, to=12, width=4, textvariable=self.n_elements,
                    command=self._rebuild_rows).pack(side="left", padx=(4, 12))

        ttk.Label(top, text="Node 0 at  X:").pack(side="left")
        ttk.Entry(top, textvariable=self.start_x, width=7).pack(side="left", padx=2)
        ttk.Label(top, text="Y:").pack(side="left")
        ttk.Entry(top, textvariable=self.start_y, width=7).pack(side="left", padx=2)

        ttk.Label(top, text="   Preset:").pack(side="left", padx=(12, 2))
        cb = ttk.Combobox(top, textvariable=self.preset, state="readonly", width=42,
                          values=list(self.PRESETS.keys()))
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", self._apply_preset)

        ttk.Label(self, text="Elements are chained: element 1 runs node 0 -> node 1, "
                             "element 2 runs node 1 -> node 2, and so on. Angles are "
                             "measured anticlockwise from the +X axis; downward loads "
                             "are negative.",
                  foreground="gray", wraplength=1000, justify="left").pack(
            anchor="w", padx=8, pady=(0, 6))

        # Scrollable body holding both tables
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8)
        self.scroll = tk.Canvas(body, highlightthickness=0)
        vsb = ttk.Scrollbar(body, orient="vertical", command=self.scroll.yview)
        hsb = ttk.Scrollbar(body, orient="horizontal", command=self.scroll.xview)
        self.scroll.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.scroll.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self.scroll)
        self.scroll.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                        lambda e: self.scroll.configure(scrollregion=self.scroll.bbox("all")))
        self.scroll.bind("<Enter>", lambda e: self.scroll.bind_all("<MouseWheel>", self._wheel))
        self.scroll.bind("<Leave>", lambda e: self.scroll.unbind_all("<MouseWheel>"))

        self.elem_frame = ttk.LabelFrame(self.inner, text="Element properties and span loads")
        self.elem_frame.pack(fill="x", pady=6)
        self.node_frame = ttk.LabelFrame(self.inner, text="Node supports and nodal point loads")
        self.node_frame.pack(fill="x", pady=6)

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=8)
        ttk.Button(bar, text="Build & Solve",
                   command=lambda: self._build(solve=True)).pack(side="left")
        ttk.Button(bar, text="Build only (send to Frame Designer)",
                   command=lambda: self._build(solve=False)).pack(side="left", padx=6)
        ttk.Button(bar, text="Reset to defaults",
                   command=self._reset).pack(side="left", padx=6)
        self.status = ttk.Label(bar, text="", foreground="gray")
        self.status.pack(side="left", padx=12)

    def _wheel(self, event):
        self.scroll.yview_scroll(int(-event.delta / 120), "units")

    # ------------------------------------------------------------------
    def _rebuild_rows(self):
        try:
            n = max(1, min(12, int(self.n_elements.get())))
        except (tk.TclError, ValueError):
            return

        old_elems = [[v.get() for v in row] for row in self.elem_vars]
        old_nodes = [{"name": r["name"].get(), "sup": r["sup"].get(),
                      "rot": r["rot"].get(),
                      "loads": [v.get() for v in r["loads"]]}
                     for r in self.node_vars]

        for w in self.elem_frame.winfo_children():
            w.destroy()
        for w in self.node_frame.winfo_children():
            w.destroy()
        self.elem_vars, self.node_vars = [], []

        # --- element table
        ttk.Label(self.elem_frame, text="Elem", width=6,
                  font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, padx=3, pady=3)
        for c, (_key, header, _default, kind) in enumerate(self.ELEM_COLS):
            ttk.Label(self.elem_frame, text=header, width=16 if kind == "dir" else 11,
                      font=("TkDefaultFont", 9, "bold")).grid(row=0, column=c + 1, padx=3, pady=3)
        for i in range(n):
            ttk.Label(self.elem_frame, text=f"E{i + 1}", width=6).grid(row=i + 1, column=0, padx=3)
            row_vars = []
            for c, (_key, _header, default, kind) in enumerate(self.ELEM_COLS):
                val = old_elems[i][c] if i < len(old_elems) else default
                var = tk.StringVar(value=val)
                if kind == "dir":
                    w = ttk.Combobox(self.elem_frame, textvariable=var, values=DIR_LABELS,
                                     state="readonly", width=15)
                elif kind == "check":
                    w = ttk.Checkbutton(self.elem_frame, variable=var,
                                        onvalue="1", offvalue="0")
                else:
                    w = ttk.Entry(self.elem_frame, textvariable=var, width=11)
                w.grid(row=i + 1, column=c + 1, padx=3, pady=1)
                row_vars.append(var)
            self.elem_vars.append(row_vars)
        ttk.Label(self.elem_frame,
                  text="UDL acts over the whole element; w1/w2 add a load varying linearly "
                       "from node i to node j; P is a point load at distance a from node i "
                       "(leave P = 0 for none). Each load has its own direction, so a rafter "
                       "can take vertical self weight and perpendicular wind at once. "
                       "'hinge i' / 'hinge j' release that member end so it carries no moment "
                       "into the joint -- other members meeting there stay rigid.",
                  foreground="gray", wraplength=1100, justify="left").grid(
            row=n + 1, column=0, columnspan=len(self.ELEM_COLS) + 1, sticky="w", padx=4, pady=(4, 2))

        # --- node table
        ttk.Label(self.node_frame, text="Q order", width=8,
                  font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, padx=3, pady=3)
        ttk.Label(self.node_frame, text="name", width=10,
                  font=("TkDefaultFont", 9, "bold")).grid(row=0, column=1, padx=3, pady=3)
        ttk.Label(self.node_frame, text="Support", width=11,
                  font=("TkDefaultFont", 9, "bold")).grid(row=0, column=2, padx=3, pady=3)
        ttk.Label(self.node_frame, text="no rotation", width=11,
                  font=("TkDefaultFont", 9, "bold")).grid(row=0, column=3, padx=3, pady=3)
        for c, (name, _default) in enumerate(self.NODE_COLS):
            ttk.Label(self.node_frame, text=name, width=11,
                      font=("TkDefaultFont", 9, "bold")).grid(row=0, column=c + 4, padx=3, pady=3)
        for i in range(n + 1):
            # row order here is the order of the global Q vector: this node
            # owns Q rows 3i, 3i+1, 3i+2
            ttk.Label(self.node_frame, text=f"{i}   ({3*i}-{3*i+2})",
                      width=10).grid(row=i + 1, column=0, padx=3)
            old = old_nodes[i] if i < len(old_nodes) else None
            name_var = tk.StringVar(value=old["name"] if old else f"N{i}")
            ttk.Entry(self.node_frame, textvariable=name_var, width=9).grid(
                row=i + 1, column=1, padx=3, pady=1)
            sup_var = tk.StringVar(value=old["sup"] if old else
                                   ("Fixed" if i in (0, n) else "Free"))
            ttk.Combobox(self.node_frame, textvariable=sup_var, values=SUPPORT_TYPES,
                         state="readonly", width=9).grid(row=i + 1, column=2, padx=3, pady=1)
            rot_var = tk.StringVar(value=old["rot"] if old else "0")
            ttk.Checkbutton(self.node_frame, variable=rot_var,
                            onvalue="1", offvalue="0").grid(row=i + 1, column=3, padx=3, pady=1)
            vs = []
            for c, (_name, default) in enumerate(self.NODE_COLS):
                val = old["loads"][c] if old else default
                var = tk.StringVar(value=val)
                ttk.Entry(self.node_frame, textvariable=var, width=11).grid(
                    row=i + 1, column=c + 4, padx=3, pady=1)
                vs.append(var)
            self.node_vars.append({"name": name_var, "sup": sup_var,
                                   "rot": rot_var, "loads": vs})
        ttk.Label(self.node_frame,
                  text="Support chooses which TRANSLATIONS are held: Roller-Y restrains "
                       "global Y (rolls horizontally), Roller-X restrains global X (rolls "
                       "vertically). 'no rotation' is separate -- tick it on a roller for a "
                       "GUIDED support that still slides but cannot rotate (so it carries a "
                       "reaction moment), or on a Free node to restrain rotation alone. "
                       "Fixed always holds rotation.",
                  foreground="gray", wraplength=1100, justify="left").grid(
            row=n + 2, column=0, columnspan=8, sticky="w", padx=4, pady=(4, 2))

    def _reset(self):
        self.elem_vars, self.node_vars = [], []
        self.n_elements.set(3)
        self.start_x.set("0")
        self.start_y.set("0")
        self.preset.set("")
        self._rebuild_rows()
        self.status.config(text="")

    def _apply_preset(self, _evt=None):
        p = self.PRESETS.get(self.preset.get())
        if not p:
            return
        self.elem_vars, self.node_vars = [], []   # drop old values, take preset wholesale
        self.n_elements.set(p["n"])
        self.start_x.set("0")
        self.start_y.set("0")
        self._rebuild_rows()
        for row_vars, spec in zip(self.elem_vars, p["elems"]):
            for var, (key, _header, default, _kind) in zip(row_vars, self.ELEM_COLS):
                var.set(spec.get(key, default))
        norot = p.get("norot") or ["0"] * len(p["supports"])
        names = p.get("names") or [f"N{i}" for i in range(len(p["supports"]))]
        for r, sup, rot, nm, nodal in zip(self.node_vars, p["supports"], norot,
                                          names, p["nodal"]):
            r["sup"].set(sup)
            r["rot"].set(rot)
            r["name"].set(nm)
            for var, val in zip(r["loads"], nodal):
                var.set(val)
        self.status.config(text="Preset loaded - press Build & Solve.")

    # ------------------------------------------------------------------
    def _build(self, solve=True):
        top = self.winfo_toplevel()
        try:
            x = float(self.start_x.get())
            y = float(self.start_y.get())
            elems = []
            for row in self.elem_vars:
                spec = {}
                for var, (key, _header, _default, kind) in zip(row, self.ELEM_COLS):
                    raw = var.get()
                    if kind == "entry":
                        spec[key] = float(raw)
                    elif kind == "dir":
                        spec[key] = dir_tag(raw)
                    else:
                        spec[key] = (raw == "1")
                elems.append(spec)
            sups = [r["sup"].get() for r in self.node_vars]
            norot = [r["rot"].get() == "1" for r in self.node_vars]
            names = [r["name"].get().strip() for r in self.node_vars]
            nodal = [[float(v.get()) for v in r["loads"]] for r in self.node_vars]
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Every numeric field must be a number "
                                 "(e.g. 200e9, -10000, 4.5).",
                                 parent=top)
            return

        for i, s in enumerate(elems):
            if s["L"] <= 0:
                messagebox.showerror("Invalid length",
                                     f"Element {i + 1}: length must be positive.", parent=top)
                return
            if s["P"] != 0 and not (0 <= s["a"] <= s["L"]):
                messagebox.showerror("Invalid point load",
                                     f"Element {i + 1}: a must be between 0 and "
                                     f"L = {s['L']:g} m.", parent=top)
                return

        nodes = [NodeItem(x, y)]
        elements = []
        for s in elems:
            rad = np.radians(s["angle"])
            x += s["L"] * np.cos(rad)
            y += s["L"] * np.sin(rad)
            nodes.append(NodeItem(x, y))
            el = ElementItem(nodes[-2], nodes[-1])
            el.E, el.A, el.I = s["E"], s["A"], s["I"]
            el.udl, el.w1, el.w2 = s["UDL"], s["w1"], s["w2"]
            el.udl_dir, el.lvl_dir = s["UDL_dir"], s["LVL_dir"]
            el.release_i, el.release_j = s["rel_i"], s["rel_j"]
            if s["P"] != 0:
                el.point_loads.append([s["P"], s["a"], s["P_dir"]])
            elements.append(el)

        for n, sup, rot, nm, (Fx, Fy, M) in zip(nodes, sups, norot, names, nodal):
            if nm:
                n.name = nm
            n.support = sup
            n.restrain_rotation = bool(rot) or sup == "Fixed"
            n.Fx, n.Fy, n.M = Fx, Fy, M

        self.designer.set_model(nodes, elements)
        self.notebook.select(0)
        if solve:
            self.designer.solve()
        self.status.config(text=f"Built {len(elements)} element(s), {len(nodes)} node(s).")


# ======================================================================
# Script Runner
# ======================================================================
class ScriptRunnerPanel(ttk.Frame):
    """Runs the original worksheet scripts in a subprocess and shows their output."""

    def __init__(self, parent):
        super().__init__(parent)
        self.proc = None
        self.out_queue = queue.Queue()
        self._build_layout()
        self._refresh_scripts()
        self.after(120, self._drain_queue)

    def _build_layout(self):
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(paned)
        paned.add(left, weight=1)
        ttk.Label(left, text="Scripts in this folder:").pack(anchor="w")
        self.script_list = tk.Listbox(left, exportselection=False)
        self.script_list.pack(fill="both", expand=True, pady=4)
        self.script_list.bind("<<ListboxSelect>>", lambda e: self._show_source())
        self.script_list.bind("<Double-Button-1>", lambda e: self._run())

        btns = ttk.Frame(left)
        btns.pack(fill="x")
        ttk.Button(btns, text="Run", command=self._run).pack(side="left")
        ttk.Button(btns, text="Stop", command=self._stop).pack(side="left", padx=4)
        ttk.Button(btns, text="Refresh", command=self._refresh_scripts).pack(side="left")
        ttk.Button(btns, text="Save output...", command=self._save_output).pack(side="left", padx=4)

        right = ttk.Frame(paned)
        paned.add(right, weight=4)
        self.sub = ttk.Notebook(right)
        self.sub.pack(fill="both", expand=True)

        out_frame = ttk.Frame(self.sub)
        src_frame = ttk.Frame(self.sub)
        self.sub.add(out_frame, text="Output")
        self.sub.add(src_frame, text="Source")

        self.output = self._make_text(out_frame)
        self.source = self._make_text(src_frame)
        self._set_text(self.output, "Select a script on the left and press Run.\n")

    @staticmethod
    def _make_text(parent):
        txt = tk.Text(parent, wrap="none", font=("Courier New", 9))
        vsb = ttk.Scrollbar(parent, orient="vertical", command=txt.yview)
        hsb = ttk.Scrollbar(parent, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set, state="disabled")
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        txt.pack(side="left", fill="both", expand=True)
        return txt

    @staticmethod
    def _set_text(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    @staticmethod
    def _append_text(widget, text):
        widget.configure(state="normal")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def _refresh_scripts(self):
        self.script_list.delete(0, "end")
        try:
            names = sorted(f for f in os.listdir(SCRIPT_DIR)
                           if f.endswith(".py") and f not in GUI_MODULES)
        except OSError:
            names = []
        for f in names:
            self.script_list.insert("end", f)

    def _selected_path(self):
        sel = self.script_list.curselection()
        if not sel:
            return None
        return os.path.join(SCRIPT_DIR, self.script_list.get(sel[0]))

    def _show_source(self):
        path = self._selected_path()
        if not path:
            return
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                self._set_text(self.source, fh.read())
        except OSError as ex:
            self._set_text(self.source, f"Could not read {path}:\n{ex}")

    def _run(self):
        top = self.winfo_toplevel()
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showinfo("Already running",
                                "A script is still running. Press Stop first.", parent=top)
            return
        path = self._selected_path()
        if not path:
            messagebox.showinfo("No script selected", "Pick a script on the left.", parent=top)
            return

        self._set_text(self.output, f"$ python \"{os.path.basename(path)}\"\n\n")
        self.sub.select(0)
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", path], cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace")
        except OSError as ex:
            self._append_text(self.output, f"Failed to start: {ex}\n")
            self.proc = None
            return

        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        try:
            for line in proc.stdout:
                self.out_queue.put(line)
        finally:
            proc.wait()
            self.out_queue.put(f"\n--- finished, exit code {proc.returncode} ---\n")

    def _drain_queue(self):
        try:
            while True:
                self._append_text(self.output, self.out_queue.get_nowait())
        except queue.Empty:
            pass
        self.after(120, self._drain_queue)

    def _stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self._append_text(self.output, "\n--- stopped by user ---\n")

    def _save_output(self):
        top = self.winfo_toplevel()
        path = filedialog.asksaveasfilename(
            parent=top, defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
            title="Save script output")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.output.get("1.0", "end-1c"))
        except OSError as ex:
            messagebox.showerror("Save failed", str(ex), parent=top)


# ======================================================================
# Help tab
# ======================================================================
HELP_TEXT = """\
EME302 STRUCTURAL ANALYSIS SUITE
================================

SIGN CONVENTIONS
  * Global X is to the right, global Y is up. Angles are measured
    anticlockwise from the +X axis.
  * Downward loads are NEGATIVE (UDL = -10000 N/m is 10 kN/m downwards).
  * Moments and rotations are positive anticlockwise.

LOAD DIRECTIONS
  Every span load -- the UDL, the varying w1/w2 load, and each point load --
  carries its own direction:

    Perp. to member   perpendicular to the member (local -y when negative).
                      This is the classical beam convention and the default.
    Global Y (vert.)  straight down (when negative) whatever angle the member
                      sits at. Use for gravity, self weight, snow.
    Global X (horiz.) horizontal whatever the member angle. Use for wind.

  On an inclined member a global-direction load has an axial component as
  well as a bending one, and both are included. On a vertical column a
  global-Y load is therefore pure axial and causes no bending at all, which
  is exactly right for self weight.

  Because directions are per load, one rafter can carry vertical self weight
  AND perpendicular wind at the same time.

  Intensities are per metre OF MEMBER, not per metre of horizontal plan. For
  a load quoted per metre of plan (snow is often given this way), multiply it
  by cos(member angle) before entering it.

NORMAL AND AXIAL PARTS OF A NON-PERPENDICULAR LOAD
  A load perpendicular to the member only bends it. A load given in a global
  direction on an inclined member also pushes ALONG it, so its equivalent
  nodal load vector has two parts:

    NORMAL (transverse)  fills the v and theta rows -- this is the bending
                         part, the one the classical beam formulas give
    AXIAL                fills the u rows -- p_ax L / 2 at each end for a
                         UDL -- and stretches or shortens the member

  Both vectors are printed separately, in local and in global axes, on the
  element tab, along with their sum. A useful check: for a vertical load the
  X components of the two parts cancel exactly in the global vectors, because
  the resultant can only be vertical.

AXIAL STRAIN
  The Results tab lists, for every element, the change in length, the axial
  strain, the same in microstrain, and the axial stress. Step 9 of each
  element tab shows the same thing worked through.

  Bending is ignored: the axial DOFs are uncoupled from the bending DOFs in
  K_local, so the strain comes straight from the two local axial
  displacements,

     dL = u_j - u_i        epsilon = dL / L        sigma = E epsilon

  both u values being in the member's own axes. (The extra shortening a bowed
  member shows is a second-order effect, outside linear theory.)

  If a member carries an axial span load -- which is what a non-perpendicular
  UDL gives it -- its internal axial force is not constant. It varies linearly
  from end to end, the strain above is the average, and the two end values of
  the force are listed underneath, tension positive.

JOINTS AND END RELEASES
  Where several members share a node they are RIGIDLY connected by default:
  they share the joint's rotation, so moment passes straight through.

  Ticking "Hinge at node i end" / "Hinge at node j end" on a member releases
  just that member's end: it gets its own rotation at the joint and carries
  no moment there, while every other member at that joint stays rigidly
  connected to the others. So a joint where three members meet can have two
  continuous and the third pinned.

  This is separate from a support. A hinge controls how members connect to
  EACH OTHER; it does not tie anything to ground. Grounding is the node's
  Support setting.

  Two things the solver checks for you:
    * If every member at a joint is hinged, that joint's rotation has no
      stiffness. It is restrained automatically instead of making the
      stiffness matrix singular, and any moment you applied there is
      reported as ignored.
    * A member hinged at BOTH ends is a two-force member -- it cannot carry
      transverse load at all. If that leaves part of the frame free to
      swing, it is reported as a mechanism rather than silently solved.

SUPPORTS
  A support is set in two independent parts: which TRANSLATIONS are held
  (the Support box) and whether the ROTATION is held (the tick box next to
  it). They are separate because a roller does not have to be free to
  rotate.

  Translations:
    Free      neither direction held
    Pinned    holds X and Y translation
    Roller-Y  holds global Y translation only -- rolls horizontally. This is
              the usual beam roller; it carries a vertical reaction.
    Roller-X  holds global X translation only -- rolls vertically, e.g. a
              column propped against a wall.
    Fixed     holds X and Y translation

  Rotation ("Restrain rotation at this node" / the "no rotation" column):
    clear     the node is free to rotate -- an ordinary pin or roller
    ticked    the node cannot rotate, so it carries a reaction MOMENT

  Useful combinations:
    Pinned   + rotation           = a Fixed support
    Roller-Y + rotation           = GUIDED support: still slides horizontally
                                    but cannot rotate. For a horizontal beam
                                    this behaves in bending like a fixed end.
    Roller-X + rotation           = GUIDED support sliding vertically. Fixed
                                    at one end and this at the other is the
                                    classic fixed-guided beam, with end
                                    moments wL^2/3 and wL^2/6 under a UDL.
    Free     + rotation           = rotation held but free to translate,
                                    which is what a line of symmetry needs.

  The support symbol drawn on the canvas is chosen from the restraints that
  are actually applied, not from the name, so it always matches the maths:
  a guided support is drawn as a rigid bar on rollers rather than as a
  triangle pinned at the node. In the reactions table, a DOF that is free
  shows "--" instead of a number, because it carries no reaction.

TABS
  Frame Designer
    Add Node mode      : click the canvas to drop a node (snapped to 0.25 m)
    Add Element mode   : click two existing nodes to connect them
    Select / Edit mode : click a node or member, edit it in the right panel,
                         then press the matching "Apply" button
    SOLVE STRUCTURE    : displacements, reactions, member end forces, peak
                         member deflections and an equilibrium check
    Save / Load Model  : store a model as JSON and reopen it later

    E0, E1, E2 ... tabs: one per element, showing every calculation that
                         element goes through, in order:
                           1  geometry, L, alpha, E, A, I, Beta
                           2  local stiffness matrix K_local
                           3  transformation matrix T from c and s
                           4  global stiffness K_hat = T^T K_local T
                           5  equivalent nodal loads, worked out SEPARATELY
                              for each UDL, LVL and point load -- the load
                              direction resolved into transverse and axial
                              parts, then each row of the 6x1 vector with
                              its formula and its value, then the sum.
                              A load that is NOT perpendicular to the member
                              gets three tables: its NORMAL (bending) part,
                              its AXIAL part, and the two added together. The
                              element totals are given the same way, as
                              f_eq_local NORMAL / AXIAL / TOTAL
                           6  the same three vectors rotated to global axes
                           6  those loads rotated to global, T^T f_eq
                           7  the assembly mapping: which global DOF each of
                              the element's six DOFs lands on (this is the
                              assembly matrix A written as a list), including
                              the extra DOF given to a hinged end
                           8  after solving: q_e, q_local = T q_e, the end
                              forces F = K_hat q_e - F_eq, and F_local = T F
                              broken out as axial, shear and moment
                           9  the axial strain, with bending ignored
                         Steps 1-7 are filled in as soon as the element
                         exists; step 8 appears once you solve. The matrices
                         are printed with a common factor pulled out, the
                         way the lab scripts print them.

    Global Q / KG tab : the assembled global system --
                           1  the DOF numbering: every global DOF with its
                              index, its label, which node and component it
                              belongs to, and whether it is free or restrained
                           2  the Q vector, split into the loads applied
                              directly at nodes, the equivalent nodal loads
                              from the element span loads, and the total
                           3  the assembled KG, labelled on rows and columns
                           4  the REDUCED system: the list of free DOFs, then
                              K_ff and Q_f. This is exactly what the lab
                              scripts assemble by hand with their A matrices
                           5  after solving: q and the reactions

    "Save full report" on the Results tab writes the results, the global
    system and every element's working out to one text file.

NAMING AND ORDERING THE DOFs
  Node names
    Each node has a Name (default N0, N1, ...). Rename it in the Node Editor
    or in the "name" column of the Beam Chain Builder. The name is used
    everywhere the node appears, and in the DOF labels: naming a node B gives
    u_B, v_B, th_B. Naming them A, B, C, D makes the output line up with a
    worksheet.

  Naming the DOFs yourself  (the DOF Names tab)
    This is the one to use when you want to say "this is q1, this is q2".
    The tab lists every global DOF with its Q row, which node and component
    it is, and whether it is free or restrained, and gives you a box to type
    your own name for it.

    Whatever you type is printed next to that row in:
      * the Q vector, KG, and the reduced K_ff / Q_f on the Global tab
      * the support reaction vector in the Results tab
      * the nodal displacements in the Results tab
      * f_eq_local and F_eq_global on each element tab
      * q_e, q_local, F_global and F_local on each element tab -- the second
        label column there is the global DOF the row assembles into, so the
        member's local force vector can be read straight against Q

    Two shortcuts:
      "Number the FREE DOFs q1, q2, ..."  numbers only the DOFs that survive
          into the reduced system, leaving the restrained ones blank. This is
          the lab worksheet convention.
      "Number ALL DOFs q1, q2, ..."       numbers every row of the full Q.
    Leave a box empty to fall back to the generated label, shown in grey
    beside it. Names are saved with the model, and they belong to the node,
    so moving a node up or down carries its names to the new rows.

  DOF label style
    Sets the GENERATED labels, used for any DOF you have not named yourself.
    The box above the Nodes list switches between
      u / v / th        u_B, v_B, th_B
      Fx / Fy / M       Fx_B, Fy_B, M_B
      q1, q2, q3 ...    straight down the vector, so q4 is simply row 3
    It applies to the Global tab, the element tabs and the results.

  DOF order
    The order of the Nodes list IS the order of the Q vector: the node in
    row k owns Q rows 3k, 3k+1 and 3k+2, shown next to each entry. Use
    Move Up / Move Down to change it. This is a relabelling only -- every
    node's displacement is unchanged, they just sit in different rows of Q.

    A hinged member end gets its own extra rotation DOF, appended after all
    the node DOFs and labelled th_E<element><end>.

  Which DOFs appear in the displacement vector
    The restrained DOFs are held at zero and drop out, so the vector that is
    actually solved contains only the free ones. They are listed explicitly
    under "free DOFs" in step 4 of the Global tab, and again at the bottom of
    the nodal displacements in the Results tab, with their index in Q.

  Beam Chain Builder
    For structures that are a chain of members (every worksheet in this
    folder is): type the element and node tables, then Build & Solve. The
    model lands in the Frame Designer tab, where it can be edited further.
    The Preset dropdown reproduces the lab worksheets, plus two that show
    the newer features: "Pitched portal" has vertical UDLs on inclined
    rafters and a hinged apex, and "Gerber beam" has an internal hinge
    between two spans.

  Script Runner
    Runs the original .py worksheet scripts unchanged and captures their
    printed matrices, so the hand-assembled results can be compared with
    the general solver.

DEFORMED SHAPE
  The deformed shape is drawn as a true curve: the cubic (Hermite) shape
  from the nodal displacements and rotations, PLUS the fixed-end deflection
  caused by the loads acting along the span. That second part is what makes
  UDL and linearly varying loads show up -- a fixed-fixed beam under a UDL
  has zero nodal displacement, so a straight line between its end nodes
  would show no deflection at all. The dotted straight line drawn with each
  member is the chord between its displaced end nodes, for comparison.
  Displacements are magnified by the "Scale" box; the loads themselves are
  not scaled.
"""


# ======================================================================
def main():
    root = tk.Tk()
    root.title("EME302 Structural Analysis Suite")
    root.geometry("1500x950")

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)

    designer = FrameDesignerPanel(nb)
    nb.add(designer, text="  Frame Designer  ")

    builder = ChainBuilderPanel(nb, designer, nb)
    nb.add(builder, text="  Beam Chain Builder  ")

    runner = ScriptRunnerPanel(nb)
    nb.add(runner, text="  Script Runner  ")

    help_frame = ttk.Frame(nb)
    nb.add(help_frame, text="  Help  ")
    help_txt = tk.Text(help_frame, wrap="word", font=("Courier New", 10))
    hsb = ttk.Scrollbar(help_frame, orient="vertical", command=help_txt.yview)
    help_txt.configure(yscrollcommand=hsb.set)
    hsb.pack(side="right", fill="y")
    help_txt.pack(side="left", fill="both", expand=True, padx=8, pady=8)
    help_txt.insert("1.0", HELP_TEXT)
    help_txt.configure(state="disabled")

    root.mainloop()


if __name__ == "__main__":
    main()
