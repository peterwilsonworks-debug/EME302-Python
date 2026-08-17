"""
Generalized 2D frame FE engine.

Supports an arbitrary number of nodes and beam elements, arbitrary supports
(Free / Pinned / Roller / Fixed) at any node, nodal point loads, UDL, point
loads at any position, and linearly varying (trapezoidal) loads on any
element. Based on the standard 2D frame (beam-column) stiffness formulation.

DOF convention: each node has 3 DOFs: [Fx, Fy, M] (u, v, theta).
Global DOF index for node i: [3*i, 3*i+1, 3*i+2]
"""

import numpy as np


class Node:
    def __init__(self, id_, x, y, support="Free"):
        self.id = id_
        self.x = x
        self.y = y
        self.support = support  # "Free", "Pinned", "Roller", "Fixed"
        self.loads = [0.0, 0.0, 0.0]  # Fx, Fy, M applied at this node

    def restrained_dofs(self):
        """Return which local dof indices (0=u,1=v,2=theta) are restrained."""
        if self.support == "Fixed":
            return [0, 1, 2]
        elif self.support in ("Pinned",):
            return [0, 1]
        elif self.support == "Roller":
            # Roller restrains vertical (global Y) translation only, by default
            return [1]
        return []


class Element:
    def __init__(self, id_, node_i, node_j, E, A, I,
                 udl=0.0, w1=0.0, w2=0.0, point_loads=None):
        """
        node_i, node_j: Node objects (local node 1 and node 2)
        udl: uniform load (N/m), acts in local -y (kept for backward compat;
             folded into w1/w2 if provided separately)
        w1, w2: linearly varying load intensity at node i / node j (N/m).
                If udl is given and w1=w2=0, treat as udl over full length.
        point_loads: list of (P, a) tuples; P downward-negative in local y,
                     a measured from node_i along the element length.
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
        self.point_loads = point_loads or []

    @property
    def L(self):
        return np.hypot(self.nj.x - self.ni.x, self.nj.y - self.ni.y)

    @property
    def alpha(self):
        return np.arctan2(self.nj.y - self.ni.y, self.nj.x - self.ni.x)

    def local_w1_w2(self):
        """Combine udl and trapezoidal w1/w2 into a single effective trapezoid."""
        return self.udl + self.w1, self.udl + self.w2

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

    def f_eq_local(self):
        """Total equivalent local nodal load vector (6x1) from UDL/trapezoid + point loads."""
        L = self.L
        w1, w2 = self.local_w1_w2()
        f = np.array([
            [0.0],
            [(L * (7*w1 + 3*w2)) / 20],
            [(L**2 * (w1/20 + w2/30))],
            [0.0],
            [(L * (3*w1 + 7*w2)) / 20],
            [-(L**2 * (w1/30 + w2/20))],
        ])
        for (P, a) in self.point_loads:
            b = L - a
            f += np.array([
                [0.0],
                [(P * b**2 * (3*a + b)) / L**3],
                [(P * a * b**2) / L**2],
                [0.0],
                [(P * a**2 * (a + 3*b)) / L**3],
                [-((P * a**2 * b) / L**2)],
            ])
        return f

    def f_eq_global(self):
        return self.T().T @ self.f_eq_local()


class Structure:
    def __init__(self, nodes, elements):
        self.nodes = nodes        # list[Node], index = node id, id in order
        self.elements = elements  # list[Element]
        self.ndof = 3 * len(nodes)

    def dof_ids(self, node):
        i = self.nodes.index(node)
        return [3*i, 3*i+1, 3*i+2]

    def assemble(self):
        n = self.ndof
        KG = np.zeros((n, n))
        QG = np.zeros((n, 1))

        # nodal loads
        for idx, node in enumerate(self.nodes):
            QG[3*idx:3*idx+3, 0] += node.loads

        # element stiffness + equivalent loads
        for el in self.elements:
            dofs = self.dof_ids(el.ni) + self.dof_ids(el.nj)
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
        return sorted(set(r))

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
                    "Check supports."
                )
            qf = np.linalg.solve(Kff, Qf)
            for i, dof in enumerate(free):
                q_full[dof, 0] = qf[i, 0]

        # reactions = KG @ q_full - QG_applied_only, but at restrained dofs
        # (nodal equivalent loads already included in QG; reactions balance total)
        R_full = KG @ q_full - QG

        # per-element end forces (global then local)
        elem_forces = {}
        for el in self.elements:
            dofs = self.dof_ids(el.ni) + self.dof_ids(el.nj)
            qe = q_full[dofs, :]
            Fg = el.K_global() @ qe - el.f_eq_global()
            # local forces (for member end shear/moment/axial)
            Fl = el.T() @ Fg
            elem_forces[el.id] = {"global": Fg, "local": Fl}

        return {
            "KG": KG, "QG": QG, "q": q_full, "R": R_full,
            "restrained": restrained, "free": free,
            "elem_forces": elem_forces,
        }