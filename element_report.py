"""
Full working for a single element, written out step by step.

Produces the same chain of quantities the lab worksheets print by hand --
local stiffness, transformation, global stiffness, the equivalent nodal loads
for each individual UDL / LVL / point load, the assembly mapping, and finally
the element's displacements and end forces -- but for whatever element the
GUI is showing.

Everything here is read from the fe_engine objects that the solver actually
used (Element.f_eq_terms(), Element.K_local(), Structure.element_dofs(), ...),
so the working shown cannot drift away from the numbers being solved.
"""

import numpy as np

from fe_engine import LOAD_DIR_LABELS

DOFS = ["u_i", "v_i", "th_i", "u_j", "v_j", "th_j"]
WIDTH = 78

# Formula text for each row of the equivalent nodal load vector, per load type
FORMULAS = {
    "UDL": ["p_ax L / 2",
            "w_t L / 2",
            "w_t L^2 / 12",
            "p_ax L / 2",
            "w_t L / 2",
            "-w_t L^2 / 12"],
    "LVL": ["L (2 p1 + p2) / 6",
            "L (7 w1 + 3 w2) / 20",
            "L^2 (w1/20 + w2/30)",
            "L (p1 + 2 p2) / 6",
            "L (3 w1 + 7 w2) / 20",
            "-L^2 (w1/30 + w2/20)"],
    "PL": ["P_ax b / L",
           "P_t b^2 (3a + b) / L^3",
           "P_t a b^2 / L^2",
           "P_ax a / L",
           "P_t a^2 (a + 3b) / L^3",
           "-P_t a^2 b / L^2"],
}


# ----------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------
def num(v, sig=6):
    """Compact scalar, e.g. 4.5, -1e+04, 0.000123."""
    v = float(v)
    if v == 0.0:
        v = 0.0          # normalise -0.0, which prints as "-0"
    return f"{v:.{sig}g}"


def _z(v):
    """Normalise -0.0, which would otherwise print as '-0.0000'."""
    v = float(v)
    return 0.0 if v == 0.0 else v


def _common_exp(M):
    """Power of ten to factor out, or 0 to print the numbers as they are."""
    M = np.asarray(M, dtype=float)
    amax = float(np.max(np.abs(M))) if M.size else 0.0
    if amax == 0.0 or not np.isfinite(amax):
        return 0
    e = int(np.floor(np.log10(amax)))
    return e if (e < -2 or e > 3) else 0


def matrix_lines(M, name, rows=None, cols=None, dec=4, colw=12):
    """
    Matrix with a common factor pulled out, in the style the lab scripts
    print (`K1 = 1 x 10^7 x ...`), with DOF labels on the rows and columns.
    """
    M = np.atleast_2d(np.asarray(M, dtype=float))
    exp = _common_exp(M)
    factor = 10.0 ** exp
    head = f"{name} ="
    if exp:
        head += f"  1 x 10^{exp}  x"
    out = [head]
    lw = max((len(r) for r in rows), default=0) + 1 if rows else 0
    if cols:
        out.append(" " * (lw + 2) + "".join(f"{c:>{colw}}" for c in cols))
    for r in range(M.shape[0]):
        lbl = f"{rows[r]:>{lw}} " if rows else ""
        vals = "".join(f"{_z(M[r, c] / factor):{colw}.{dec}f}"
                       for c in range(M.shape[1]))
        out.append(f"{lbl} [{vals} ]")
    return out


def vector_lines(v, name, rows=DOFS, dec=4, colw=14):
    v = np.asarray(v, dtype=float).reshape(-1)
    exp = _common_exp(v)
    factor = 10.0 ** exp
    head = f"{name} ="
    if exp:
        head += f"  1 x 10^{exp}  x"
    out = [head]
    lw = max((len(r) for r in rows), default=0) + 1 if rows else 0
    for i, val in enumerate(v):
        lbl = f"{rows[i]:>{lw}} " if rows else ""
        out.append(f"{lbl} [{_z(val / factor):{colw}.{dec}f} ]")
    return out


def step(n, title):
    return ["", "-" * WIDTH, f"STEP {n}  --  {title}", "-" * WIDTH]


def dir_name(tag):
    return LOAD_DIR_LABELS.get(tag, tag)


# ----------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------
def element_report(el, struct=None, result=None, title=None):
    """
    el:     fe_engine.Element
    struct: fe_engine.Structure it belongs to (for the assembly step)
    result: the dict from Structure.solve() (for the final step)
    """
    L = el.L
    alpha = el.alpha
    EI = el.E * el.I
    c, s = np.cos(alpha), np.sin(alpha)
    out = []

    out.append("=" * WIDTH)
    out.append(title or f"ELEMENT E{el.id}   (node {el.ni.id} -> node {el.nj.id})")
    out.append("=" * WIDTH)

    # ---------------------------------------------------------------- 1
    out += step(1, "GEOMETRY AND SECTION PROPERTIES")
    out.append(f"   node i = N{el.ni.id} at ({num(el.ni.x)}, {num(el.ni.y)}) m")
    out.append(f"   node j = N{el.nj.id} at ({num(el.nj.x)}, {num(el.nj.y)}) m")
    dx, dy = el.nj.x - el.ni.x, el.nj.y - el.ni.y
    out.append("")
    out.append(f"   dx = xj - xi = {num(dx)} m        dy = yj - yi = {num(dy)} m")
    out.append(f"   L     = sqrt(dx^2 + dy^2)  = {num(L)} m")
    out.append(f"   alpha = atan2(dy, dx)      = {num(alpha)} rad "
               f"= {num(np.degrees(alpha))} deg")
    out.append("")
    out.append(f"   E = {num(el.E)} Pa      A = {num(el.A)} m^2      I = {num(el.I)} m^4")
    out.append(f"   EI = {num(EI)} N m^2      EA = {num(el.E * el.A)} N")
    if el.I != 0:
        out.append(f"   Beta = A L^2 / I = {num(el.A)} x {num(L)}^2 / {num(el.I)} "
                   f"= {num((el.A * L**2) / el.I)}")
    if el.release_i or el.release_j:
        out.append("")
        ends = []
        if el.release_i:
            ends.append("node i end")
        if el.release_j:
            ends.append("node j end")
        out.append(f"   End releases: {' and '.join(ends)} HINGED -- this member carries")
        out.append("   no moment there and its rotation is independent of the joint's.")

    # ---------------------------------------------------------------- 2
    out += step(2, "LOCAL STIFFNESS MATRIX  K_local")
    out.append("   In local axes (x along the member, y perpendicular):")
    out.append("")
    out.append("                     [ Beta    0      0     -Beta    0      0    ]")
    out.append("                     [  0     12    6L        0    -12    6L     ]")
    out.append("        E I          [  0     6L    4L^2      0    -6L    2L^2   ]")
    out.append("   K = -----   x     [ -Beta   0      0      Beta    0      0    ]")
    out.append("        L^3          [  0    -12   -6L        0     12   -6L     ]")
    out.append("                     [  0     6L    2L^2      0    -6L    4L^2   ]")
    out.append("")
    out.append(f"   E I / L^3 = {num(EI)} / {num(L)}^3 = {num(EI / L**3)}")
    out.append("")
    out += ["   " + ln for ln in matrix_lines(el.K_local(), "K_local", DOFS, DOFS)]

    # ---------------------------------------------------------------- 3
    out += step(3, "TRANSFORMATION MATRIX  T")
    out.append(f"   c = cos(alpha) = {num(c)}        s = sin(alpha) = {num(s)}")
    out.append("")
    out.append("            [  c   s   0 ]                [ lambda    0    ]")
    out.append("   lambda = [ -s   c   0 ]          T =   [   0    lambda  ]")
    out.append("            [  0   0   1 ]")
    out.append("")
    out += ["   " + ln for ln in matrix_lines(el.T(), "T", DOFS, DOFS)]

    # ---------------------------------------------------------------- 4
    out += step(4, "ELEMENT STIFFNESS IN GLOBAL AXES  K_hat = T^T K_local T")
    out.append("   This is the matrix that gets added into the global stiffness")
    out.append("   matrix KG. Its DOFs are global (Fx, Fy, M) at each node.")
    out.append("")
    out += ["   " + ln for ln in matrix_lines(el.K_global(), "K_hat", DOFS, DOFS)]

    # ---------------------------------------------------------------- 5
    out += step(5, "EQUIVALENT NODAL LOADS FROM THE SPAN LOADS")
    terms = el.f_eq_terms()
    if not terms:
        out.append("   No UDL, LVL or point load on this element, so f_eq = 0.")
    else:
        out.append("   Each span load is replaced by the nodal forces and moments that")
        out.append("   would hold a fixed-fixed member in the same shape.")
        for k, t in enumerate(terms, start=1):
            out.append("")
            out.append(f"   {'.' * (WIDTH - 6)}")
            out += _term_lines(el, t, k)

        out.append("")
        out.append("   Sum of all the above:")
        out.append("")
        out += ["   " + ln for ln in vector_lines(el.f_eq_local(), "f_eq_local")]

    # ---------------------------------------------------------------- 6
    out += step(6, "EQUIVALENT NODAL LOADS IN GLOBAL AXES  F_eq = T^T f_eq_local")
    out.append("   Rotated into global axes, ready to be added into the global load")
    out.append("   vector QG at this element's DOFs.")
    out.append("")
    out += ["   " + ln for ln in vector_lines(el.f_eq_global(), "F_eq_global")]

    # ---------------------------------------------------------------- 7
    out += step(7, "ASSEMBLY  --  where this element goes in KG and QG")
    if struct is None:
        out.append("   (build the model to see the assembly mapping)")
    else:
        dofs = struct.element_dofs(el)
        out.append("   Each local DOF of this element maps to one global DOF. This is the")
        out.append("   assembly matrix A written as a list: KG[dofs, dofs] += K_hat, and")
        out.append("   QG[dofs] += F_eq_global.")
        out.append("")
        out.append("      element DOF        global DOF     belongs to")
        out.append("      " + "-" * 56)
        for i, (name, g) in enumerate(zip(DOFS, dofs)):
            if g < struct.n_node_dofs:
                node_no, comp = divmod(g, 3)
                nid = struct.nodes[node_no].id
                who = f"node N{nid}, {'Fx Fy M'.split()[comp]}"
            else:
                end = "i" if i < 3 else "j"
                who = f"THIS MEMBER ONLY (hinge at node {end})"
            out.append(f"      {name:<8} ({i})  ->   {g:>5}          {who}")
        if el.release_i or el.release_j:
            out.append("")
            out.append("   A released end keeps the joint's two TRANSLATION DOFs, so the")
            out.append("   hinge still transmits force; only the rotation is given its own")
            out.append("   DOF, which is why no moment can pass through it.")

    # ---------------------------------------------------------------- 8
    out += step(8, "ELEMENT DISPLACEMENTS AND END FORCES")
    if struct is None or result is None:
        out.append("   Press SOLVE STRUCTURE to fill this in.")
    else:
        dofs = struct.element_dofs(el)
        qe = result["q"][dofs, :]
        ql = el.T() @ qe
        Fg = el.K_global() @ qe - el.f_eq_global()
        Fl = el.T() @ Fg
        out.append("   Taken from the solved global displacement vector q:")
        out.append("")
        out += ["   " + ln for ln in vector_lines(qe, "q_e (global axes, m and rad)",
                                                  DOFS, dec=8)]
        out.append("")
        out.append("   Rotated into the member's own axes,  q_local = T q_e :")
        out.append("")
        out += ["   " + ln for ln in vector_lines(ql, "q_local", DOFS, dec=8)]
        out.append("")
        out.append("   End forces in global axes,  F = K_hat q_e - F_eq_global :")
        out.append("")
        out += ["   " + ln for ln in vector_lines(Fg, "F_global", DOFS)]
        out.append("")
        out.append("   End forces in local axes,  F_local = T F  --  these are the")
        out.append("   axial force, shear and bending moment at each end:")
        out.append("")
        out += ["   " + ln for ln in vector_lines(Fl, "F_local",
                                                  ["N_i", "V_i", "M_i",
                                                   "N_j", "V_j", "M_j"])]
        f = Fl.ravel()
        out.append("")
        out.append(f"      End i:  axial N = {f[0]:12.2f} N   shear V = {f[1]:12.2f} N"
                   f"   moment M = {f[2]:12.2f} Nm")
        out.append(f"      End j:  axial N = {f[3]:12.2f} N   shear V = {f[4]:12.2f} N"
                   f"   moment M = {f[5]:12.2f} Nm")
        if el.release_i:
            out.append(f"      (node i end is hinged, so M_i = {f[2]:.3e} Nm, i.e. zero)")
        if el.release_j:
            out.append(f"      (node j end is hinged, so M_j = {f[5]:.3e} Nm, i.e. zero)")

    out.append("")
    return "\n".join(out)


def _term_lines(el, t, k):
    """The working for one individual span load."""
    L = el.L
    out = []
    kind = t["kind"]
    inp = t["inputs"]

    if kind == "UDL":
        out.append(f"   Load {k}:  UDL of w = {num(inp['w'])} N/m over the whole member")
    elif kind == "LVL":
        out.append(f"   Load {k}:  linearly varying load, w1 = {num(inp['w1'])} N/m at "
                   f"node i")
        out.append(f"             to w2 = {num(inp['w2'])} N/m at node j")
    else:
        out.append(f"   Load {k}:  point load P = {num(inp['P'])} N at a = {num(inp['a'])} m "
                   f"from node i")
        out.append(f"             (so b = L - a = {num(inp['b'])} m)")

    out.append(f"             acting: {dir_name(t['direction'])}")
    out.append("")

    # direction resolution
    if t["direction"] == "local":
        out.append("      The load is already perpendicular to the member, so it is")
        out.append("      entirely transverse and has no axial component:")
        out.append("         transverse factor t = 1        axial factor n = 0")
    else:
        g = "gx = 1, gy = 0" if t["direction"] == "gx" else "gx = 0, gy = 1"
        out.append(f"      Resolve the global direction ({g}) onto the member's own")
        out.append("      axes, using c = cos(alpha) and s = sin(alpha):")
        out.append(f"         transverse  t = -gx s + gy c = {num(t['t'])}")
        out.append(f"         axial       n =  gx c + gy s = {num(t['ax'])}")
    out.append("")

    if kind == "PL":
        out.append(f"      P_t  = P x t = {num(inp['P'])} x {num(t['t'])} = {num(t['P_t'])} N")
        out.append(f"      P_ax = P x n = {num(inp['P'])} x {num(t['ax'])} = {num(t['P_ax'])} N")
    elif kind == "UDL":
        out.append(f"      w_t  = w x t = {num(t['w1_t'])} N/m   (bending)")
        out.append(f"      p_ax = w x n = {num(t['p1_ax'])} N/m   (axial)")
    else:
        out.append(f"      w1 = {num(t['w1_t'])} N/m, w2 = {num(t['w2_t'])} N/m   (bending)")
        out.append(f"      p1 = {num(t['p1_ax'])} N/m, p2 = {num(t['p2_ax'])} N/m   (axial)")
    out.append("")

    out.append("      local DOF      formula                        value (N or Nm)")
    out.append("      " + "-" * 64)
    f = t["f"].ravel()
    for i, (dof, formula) in enumerate(zip(DOFS, FORMULAS[kind])):
        out.append(f"      f{i + 1} ({dof:<4}) = {formula:<28} = {_z(f[i]):16.4f}")
    return out


def full_report(struct, elements, result=None, header=None):
    """Every element's working, one after another, for saving to a file."""
    parts = []
    if header:
        parts.append(header)
    for el in elements:
        parts.append(element_report(el, struct, result))
    return "\n\n".join(parts)
