"""
Generalized 2D frame FE engine.

Supports an arbitrary number of nodes and beam elements, arbitrary supports
(Free / Pinned / Roller / Fixed) at any node, nodal point loads, UDL, point
loads at any position, and linearly varying (trapezoidal) loads on any
element. Based on the standard 2D frame (beam-column) stiffness formulation.

DOF convention: each node has 3 DOFs: [Fx, Fy, M] (u, v, theta).
Global DOF index for node i: [3*i, 3*i+1, 3*i+2]

Two extensions beyond the plain formulation:

  * END RELEASES. A member end can be released (hinged) so that it does not
    share the joint's rotation. Instead of condensing the element matrix, the
    released end is given its OWN extra rotation DOF, appended after the node
    DOFs -- i.e. the assembly matrix simply maps that element's rotation
    somewhere else. Translations stay tied, so the joint still transmits
    force. This handles any mix of rigid and hinged members at one joint.

  * LOAD DIRECTIONS. Distributed and point loads on a member can act
    perpendicular to it (the usual local convention) or along global X or
    global Y, so a vertical load can be put on an inclined member. A global
    direction generally has an axial component as well as a transverse one,
    so the equivalent nodal load vector carries axial terms too.
"""

import numpy as np

# Load direction tags used by Element.udl_dir / lvl_dir / point load entries
LOAD_DIRS = ("local", "gx", "gy")
LOAD_DIR_LABELS = {"local": "Perp. to member",
                   "gx": "Global X (horiz.)",
                   "gy": "Global Y (vert.)"}


class Node:
    def __init__(self, id_, x, y, support="Free", roller_axis="Y"):
        """
        support: "Free", "Pinned", "Roller", "Fixed"
        roller_axis: only used when support == "Roller". "Y" restrains global Y
                     translation (roller rolls horizontally, the usual case);
                     "X" restrains global X translation (roller on a vertical
                     face, rolls up and down).
        """
        self.id = id_
        self.x = x
        self.y = y
        self.support = support
        self.roller_axis = roller_axis
        self.loads = [0.0, 0.0, 0.0]  # Fx, Fy, M applied at this node

    def restrained_dofs(self):
        """Return which local dof indices (0=u,1=v,2=theta) are restrained."""
        if self.support == "Fixed":
            return [0, 1, 2]
        elif self.support in ("Pinned",):
            return [0, 1]
        elif self.support == "Roller":
            # Restrains translation along the chosen global axis only
            return [0] if str(self.roller_axis).upper().startswith("X") else [1]
        return []


class Element:
    def __init__(self, id_, node_i, node_j, E, A, I,
                 udl=0.0, w1=0.0, w2=0.0, point_loads=None,
                 udl_dir="local", lvl_dir="local",
                 release_i=False, release_j=False):
        """
        node_i, node_j: Node objects (local node 1 and node 2)
        udl: uniform load intensity (N/m) over the whole element
        w1, w2: linearly varying load intensity at node i / node j (N/m),
                added on top of udl
        point_loads: list of (P, a) or (P, a, direction) tuples; a is measured
                     from node_i along the element length
        udl_dir, lvl_dir: direction of the uniform / varying load, one of
                     "local" (perpendicular to the member, downward negative),
                     "gx" (along global X), "gy" (along global Y). Each load
                     carries its own direction, so e.g. a rafter can take
                     vertical self weight and perpendicular wind at once.
                     Intensities are per metre OF MEMBER, not per metre of
                     horizontal projection.
        release_i, release_j: True to hinge that end -- the member's rotation
                     at that end is independent of the joint's rotation, so it
                     carries no moment there.
        """
        self.id = id_
        self.ni = node_i
        self.nj = node_j
        self.E = E
        self.A = A
        self.I = I
        self.udl = udl
        self.w1 = w1
        self.w2 = w2
        self.udl_dir = udl_dir
        self.lvl_dir = lvl_dir
        self.release_i = bool(release_i)
        self.release_j = bool(release_j)
        self.point_loads = [self._norm_point_load(p) for p in (point_loads or [])]

    @staticmethod
    def _norm_point_load(p):
        """Accept (P, a) or (P, a, direction); direction defaults to local."""
        if len(p) >= 3:
            return (float(p[0]), float(p[1]), str(p[2]))
        return (float(p[0]), float(p[1]), "local")

    @property
    def L(self):
        return np.hypot(self.nj.x - self.ni.x, self.nj.y - self.ni.y)

    @property
    def alpha(self):
        return np.arctan2(self.nj.y - self.ni.y, self.nj.x - self.ni.x)

    # ------------------------------------------------------------------
    # Load direction handling
    # ------------------------------------------------------------------
    def dir_components(self, direction):
        """
        Resolve a unit load acting in `direction` into (transverse, axial)
        components in the member's local axes.

        "local" means the load is already perpendicular to the member, so it
        is purely transverse. A global direction is projected onto the local
        +y (transverse) and +x (axial) unit vectors.
        """
        if direction == "local":
            return 1.0, 0.0
        c, s = np.cos(self.alpha), np.sin(self.alpha)
        gx, gy = (1.0, 0.0) if direction == "gx" else (0.0, 1.0)
        return (-gx * s + gy * c), (gx * c + gy * s)

    def dir_vector(self, direction):
        """Global (dx, dy) unit vector a positive load in `direction` acts along."""
        if direction == "local":
            # local +y, i.e. perpendicular to the member
            return (-np.sin(self.alpha), np.cos(self.alpha))
        return (1.0, 0.0) if direction == "gx" else (0.0, 1.0)

    def load_components(self):
        """
        Combine the uniform and varying loads into a single equivalent local
        trapezoid, returning (w1_t, w2_t, p1_ax, p2_ax): transverse intensity
        at each end, and axial intensity at each end.
        """
        t_udl, a_udl = self.dir_components(self.udl_dir)
        t_lvl, a_lvl = self.dir_components(self.lvl_dir)
        w1_t = self.udl * t_udl + self.w1 * t_lvl
        w2_t = self.udl * t_udl + self.w2 * t_lvl
        p1_a = self.udl * a_udl + self.w1 * a_lvl
        p2_a = self.udl * a_udl + self.w2 * a_lvl
        return w1_t, w2_t, p1_a, p2_a

    def local_w1_w2(self):
        """Transverse (bending) part of the distributed load, at each end."""
        w1_t, w2_t, _, _ = self.load_components()
        return w1_t, w2_t

    def resultant_global(self):
        """Total (Fx, Fy) of this member's span loads, in global axes."""
        L = self.L
        Fx = Fy = 0.0
        for mag, direction in ((self.udl * L, self.udl_dir),
                               ((self.w1 + self.w2) * L / 2.0, self.lvl_dir)):
            dx, dy = self.dir_vector(direction)
            Fx += mag * dx
            Fy += mag * dy
        for (P, _a, d) in self.point_loads:
            dx, dy = self.dir_vector(d)
            Fx += P * dx
            Fy += P * dy
        return Fx, Fy

    # ------------------------------------------------------------------
    # Stiffness
    # ------------------------------------------------------------------
    def K_local(self):
        E, A, L, I = self.E, self.A, self.L, self.I
        Beta = (A * L**2) / I
        K = np.array([
            [Beta, 0, 0, -Beta, 0, 0],
            [0, 12, 6*L, 0, -12, 6*L],
            [0, 6*L, 4*L**2, 0, -6*L, 2*L**2],
            [-Beta, 0, 0, Beta, 0, 0],
            [0, -12, -6*L, 0, 12, -6*L],
            [0, 6*L, 2*L**2, 0, -6*L, 4*L**2],
        ])
        return ((E * I) / L**3) * K

    def T(self):
        c, s = np.cos(self.alpha), np.sin(self.alpha)
        lam = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        Tm = np.zeros((6, 6))
        Tm[:3, :3] = lam
        Tm[3:, 3:] = lam
        return Tm

    def K_global(self):
        T = self.T()
        return T.T @ self.K_local() @ T

    # ------------------------------------------------------------------
    # Equivalent nodal loads
    # ------------------------------------------------------------------
    def f_eq_terms(self):
        """
        The equivalent nodal load vector broken down into one entry per
        applied span load, so the working can be shown load by load.

        Each entry is a dict with the load's description, how its direction
        resolved into local transverse/axial intensities, and its own 6x1
        contribution. f_eq_local() is exactly the sum of these, so anything
        displayed from here is the same arithmetic the solver used.
        """
        L = self.L
        terms = []

        if self.udl != 0:
            t, ax = self.dir_components(self.udl_dir)
            wt, wa = self.udl * t, self.udl * ax
            terms.append({
                "kind": "UDL", "direction": self.udl_dir,
                "inputs": {"w": self.udl},
                "t": t, "ax": ax, "w1_t": wt, "w2_t": wt, "p1_ax": wa, "p2_ax": wa,
                "f": np.array([
                    [wa * L / 2],
                    [wt * L / 2],
                    [wt * L**2 / 12],
                    [wa * L / 2],
                    [wt * L / 2],
                    [-wt * L**2 / 12],
                ]),
            })

        if self.w1 != 0 or self.w2 != 0:
            t, ax = self.dir_components(self.lvl_dir)
            w1t, w2t = self.w1 * t, self.w2 * t
            p1, p2 = self.w1 * ax, self.w2 * ax
            terms.append({
                "kind": "LVL", "direction": self.lvl_dir,
                "inputs": {"w1": self.w1, "w2": self.w2},
                "t": t, "ax": ax, "w1_t": w1t, "w2_t": w2t, "p1_ax": p1, "p2_ax": p2,
                "f": np.array([
                    [(L * (2*p1 + p2)) / 6],
                    [(L * (7*w1t + 3*w2t)) / 20],
                    [(L**2 * (w1t/20 + w2t/30))],
                    [(L * (p1 + 2*p2)) / 6],
                    [(L * (3*w1t + 7*w2t)) / 20],
                    [-(L**2 * (w1t/30 + w2t/20))],
                ]),
            })

        for k, (P, a, d) in enumerate(self.point_loads):
            t, ax = self.dir_components(d)
            Pt, Pa = P * t, P * ax
            b = L - a
            terms.append({
                "kind": "PL", "direction": d, "index": k,
                "inputs": {"P": P, "a": a, "b": b},
                "t": t, "ax": ax, "P_t": Pt, "P_ax": Pa,
                "f": np.array([
                    [Pa * b / L],
                    [(Pt * b**2 * (3*a + b)) / L**3],
                    [(Pt * a * b**2) / L**2],
                    [Pa * a / L],
                    [(Pt * a**2 * (a + 3*b)) / L**3],
                    [-((Pt * a**2 * b) / L**2)],
                ]),
            })

        return terms

    def f_eq_local(self):
        """Total equivalent local nodal load vector (6x1) from the span loads."""
        f = np.zeros((6, 1))
        for term in self.f_eq_terms():
            f = f + term["f"]
        return f

    def f_eq_global(self):
        return self.T().T @ self.f_eq_local()

    # ------------------------------------------------------------------
    # Deflected shape
    # ------------------------------------------------------------------
    def particular_deflection(self, x):
        """
        Local transverse deflection v_p(x) of a FIXED-FIXED member under the
        span loads (UDL, linearly varying load, point loads).

        Why this is needed: the assembly replaces span loads by equivalent
        nodal loads, so the nodal displacements alone describe only the
        homogeneous (cubic) part of the deflection. A member whose ends barely
        move -- e.g. a fixed-fixed beam under a UDL -- then appears perfectly
        straight unless this fixed-end particular solution is added back.

        Only the TRANSVERSE (bending) part of the load is included. The axial
        stretch between the end nodes is left linear: for realistic EA it is
        orders of magnitude smaller than the bending deflection and invisible
        at any sensible drawing scale. Forces and reactions are unaffected --
        the axial equivalent nodal loads are exact.

        Sign convention: loads and v_p are in local +y (downward loads are
        negative for a horizontal member), consistent with f_eq_local().
        """
        x = np.asarray(x, dtype=float)
        L = self.L
        EI = self.E * self.I
        vp = np.zeros_like(x)
        if L <= 0 or EI == 0:
            return vp

        w1, w2, _, _ = self.load_components()

        # Uniform part: v = w x^2 (L-x)^2 / (24 EI)   (max wL^4/384EI at mid)
        if w1 != 0.0:
            vp = vp + w1 * x**2 * (L - x)**2 / (24.0 * EI)

        # Ramp part (0 at node i -> d at node j), from EI v'''' = d x / L
        d = w2 - w1
        if d != 0.0:
            vp = vp + d * (x**5 / (120.0 * L)
                           - L * x**3 / 40.0
                           + L**2 * x**2 / 60.0) / EI

        # Point loads: standard fixed-fixed solution, transverse part at x = a
        for (P, a, direction) in self.point_loads:
            Pt = P * self.dir_components(direction)[0]
            if Pt == 0.0:
                continue
            a = float(np.clip(a, 0.0, L))
            b = L - a
            xm = L - x  # mirrored coordinate, measured from node j
            left = Pt * b**2 * x**2 * (3*a*L - 3*a*x - b*x) / (6.0 * EI * L**3)
            right = Pt * a**2 * xm**2 * (3*b*L - 3*b*xm - a*xm) / (6.0 * EI * L**3)
            vp = vp + np.where(x <= a, left, right)

        return vp

    def deflected_shape(self, qe_global, npts=41, scale=1.0):
        """
        Global (X, Y) polyline of the deformed member centreline.

        qe_global: 6x1 (or length-6) global displacement vector for this
                   element's DOFs, ordered [ui, vi, thi, uj, vj, thj]. At a
                   released end the rotation is that member's OWN rotation,
                   so a hinge shows up as a kink between adjacent members.
        scale:     visual magnification applied to the displacements only.
        """
        ql = (self.T() @ np.asarray(qe_global, dtype=float).reshape(6, 1)).ravel()
        L = self.L
        x = np.linspace(0.0, L, npts)
        if L <= 0:
            return np.full(npts, self.ni.x), np.full(npts, self.ni.y)
        xi = x / L

        # Axial: linear between end displacements
        u = ql[0] + (ql[3] - ql[0]) * xi

        # Transverse: Hermite cubic from end DOFs (homogeneous part) ...
        H1 = 1 - 3*xi**2 + 2*xi**3
        H2 = L * (xi - 2*xi**2 + xi**3)
        H3 = 3*xi**2 - 2*xi**3
        H4 = L * (-xi**2 + xi**3)
        v = H1*ql[1] + H2*ql[2] + H3*ql[4] + H4*ql[5]

        # ... plus the fixed-end deflection caused by the span loads
        v = v + self.particular_deflection(x)

        c, s = np.cos(self.alpha), np.sin(self.alpha)
        X0 = self.ni.x + x * c
        Y0 = self.ni.y + x * s
        X = X0 + scale * (u * c - v * s)
        Y = Y0 + scale * (u * s + v * c)
        return X, Y


class Structure:
    def __init__(self, nodes, elements):
        self.nodes = list(nodes)        # index = node number
        self.elements = list(elements)
        self._index = {id(n): i for i, n in enumerate(self.nodes)}
        self.n_node_dofs = 3 * len(self.nodes)
        self._build_dof_map()

    # ------------------------------------------------------------------
    # DOF mapping (this is the assembly matrix, built explicitly)
    # ------------------------------------------------------------------
    def dof_ids(self, node):
        i = self._index[id(node)]
        return [3*i, 3*i+1, 3*i+2]

    def _build_dof_map(self):
        """
        Assign global DOF numbers to every element end.

        A rigidly connected end uses the joint's own three DOFs. A released
        end keeps the joint's two translation DOFs -- a hinge still transmits
        force -- but gets a fresh rotation DOF of its own, appended after the
        node DOFs, so its rotation is independent of everything else at that
        joint.
        """
        nxt = self.n_node_dofs
        self._elem_dofs = []
        self.release_dofs = {}   # (element index, "i"|"j") -> extra dof number
        for k, el in enumerate(self.elements):
            di = self.dof_ids(el.ni)
            dj = self.dof_ids(el.nj)
            if getattr(el, "release_i", False):
                di = di[:2] + [nxt]
                self.release_dofs[(k, "i")] = nxt
                nxt += 1
            if getattr(el, "release_j", False):
                dj = dj[:2] + [nxt]
                self.release_dofs[(k, "j")] = nxt
                nxt += 1
            self._elem_dofs.append(di + dj)
        self.ndof = nxt

    def element_dofs(self, el):
        for k, e in enumerate(self.elements):
            if e is el:
                return self._elem_dofs[k]
        raise KeyError("element not part of this structure")

    def unconnected_rotation_dofs(self):
        """
        Joint rotation DOFs that no member is rigidly attached to.

        If every member at a joint is hinged (or no member reaches it at all)
        the joint's own rotation has no stiffness. It is not a physical
        freedom, so it is restrained rather than left to make the stiffness
        matrix singular.
        """
        used = set()
        for el in self.elements:
            if not getattr(el, "release_i", False):
                used.add(self.dof_ids(el.ni)[2])
            if not getattr(el, "release_j", False):
                used.add(self.dof_ids(el.nj)[2])
        return [3*i + 2 for i in range(len(self.nodes)) if 3*i + 2 not in used]

    def deflected_shape(self, el, q_full, npts=41, scale=1.0):
        """Convenience: global (X, Y) polyline of one element's deformed shape."""
        return el.deflected_shape(q_full[self.element_dofs(el), :],
                                  npts=npts, scale=scale)

    # ------------------------------------------------------------------
    def assemble(self):
        n = self.ndof
        KG = np.zeros((n, n))
        QG = np.zeros((n, 1))

        # nodal loads
        for idx, node in enumerate(self.nodes):
            QG[3*idx:3*idx+3, 0] += node.loads

        # element stiffness + equivalent loads
        for k, el in enumerate(self.elements):
            dofs = self._elem_dofs[k]
            Kg = el.K_global()
            Fg = el.f_eq_global()
            for a in range(6):
                QG[dofs[a], 0] += Fg[a, 0]
                for b in range(6):
                    KG[dofs[a], dofs[b]] += Kg[a, b]

        return KG, QG

    def restrained_global_dofs(self):
        r = []
        for idx, node in enumerate(self.nodes):
            for local_dof in node.restrained_dofs():
                r.append(3*idx + local_dof)
        r.extend(self.unconnected_rotation_dofs())
        return sorted(set(r))

    def warnings(self):
        """Modelling problems worth telling the user about, as plain strings."""
        msgs = []
        dead_rot = set(self.unconnected_rotation_dofs())
        for idx, node in enumerate(self.nodes):
            if 3*idx + 2 in dead_rot and node.loads[2] != 0:
                msgs.append(
                    f"Node {node.id}: a moment of {node.loads[2]:g} Nm is applied, but every "
                    f"member at that joint is hinged, so there is nothing for it to act on. "
                    f"The moment has been ignored -- make one member end rigid to carry it.")
        return msgs

    def solve(self):
        KG, QG = self.assemble()
        n = self.ndof
        restrained = self.restrained_global_dofs()
        free = [i for i in range(n) if i not in restrained]

        q_full = np.zeros((n, 1))

        if free:
            Kff = KG[np.ix_(free, free)]
            Qf = QG[free, :]
            # subtract contribution from restrained dofs (all zero displacement,
            # so no term needed unless support settlement is modeled)
            if np.linalg.matrix_rank(Kff) < len(free):
                raise np.linalg.LinAlgError(
                    "Structure is unstable / mechanism detected (singular stiffness matrix). "
                    "Check supports, and check that hinges have not released so many member "
                    "ends that part of the frame can swing freely."
                )
            qf = np.linalg.solve(Kff, Qf)
            for i, dof in enumerate(free):
                q_full[dof, 0] = qf[i, 0]

        # reactions = KG @ q_full - QG_applied_only, but at restrained dofs
        # (nodal equivalent loads already included in QG; reactions balance total)
        R_full = KG @ q_full - QG

        # per-element end forces (global then local)
        elem_forces = {}
        for k, el in enumerate(self.elements):
            qe = q_full[self._elem_dofs[k], :]
            Fg = el.K_global() @ qe - el.f_eq_global()
            # local forces (for member end shear/moment/axial)
            Fl = el.T() @ Fg
            elem_forces[el.id] = {"global": Fg, "local": Fl}

        return {
            "KG": KG, "QG": QG, "q": q_full, "R": R_full,
            "restrained": restrained, "free": free,
            "elem_forces": elem_forces,
            "warnings": self.warnings(),
        }
