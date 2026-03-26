from flask import Flask, request, jsonify, Response
from scipy.optimize import linprog
import numpy as np
import math, io, base64
import os

app = Flask(__name__)

def read_html():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

@app.route("/")
def index():
    return Response(read_html(), content_type="text/html; charset=utf-8")

@app.route("/solve", methods=["POST"])
def solve():
    data = request.json
    try:
        obj_type  = data.get("obj_type", "max")
        c         = np.array(data["c"], dtype=float)
        A         = np.array(data["A"], dtype=float)
        b         = np.array(data["b"], dtype=float)
        # Each element: "le" (<=), "ge" (>=), or "eq" (=)
        con_types = data.get("constraint_types",
                             ["le" if obj_type == "max" else "ge"] * len(b))
        n_vars = len(c)
        n_con  = len(b)
        steps  = []

        # ── STEP 1: Show original problem ──────────────────────────────
        steps.append({"title": "Step 1: Original Problem",
                       "body": build_problem_text(c, A, b, obj_type, n_vars, n_con, con_types)})

        # ── STEP 2: Handle negative RHS ────────────────────────────────
        # Multiply any row with b<0 by -1; inequality flips for le/ge
        A_std = A.copy(); b_std = b.copy(); ct_std = list(con_types)
        flipped = []
        for i in range(n_con):
            if b_std[i] < 0:
                A_std[i] *= -1; b_std[i] *= -1
                if ct_std[i] == "le":
                    ct_std[i] = "ge"
                elif ct_std[i] == "ge":
                    ct_std[i] = "le"
                # equality stays equality
                flipped.append(i+1)
        if flipped:
            details = []
            for idx in [x-1 for x in flipped]:
                orig = sign_html(con_types[idx])
                new  = sign_html(ct_std[idx])
                details.append(f"Row {idx+1}: multiply by &minus;1 &rArr; {orig} becomes {new}")
            steps.append({"title": "Step 2: Handle Negative RHS",
                           "body": (f"Constraints {flipped} have negative b<sub>i</sub>.<br>"
                                    "Multiply both sides by &minus;1 (reverses &le;/&ge;, keeps =):<br>"
                                    + "<br>".join(details))})
        else:
            steps.append({"title": "Step 2: Check RHS Values",
                           "body": "All RHS values are non-negative. No adjustment needed."})

        # ── STEP 3: Convert to linprog standard form ───────────────────
        # linprog: MIN c_lp.x  s.t.  A_ub x <= b_ub,  A_eq x = b_eq
        A_ub_rows, b_ub_rows = [], []
        A_eq_rows, b_eq_rows = [], []
        for i in range(n_con):
            ai, bi, ct = A_std[i], b_std[i], ct_std[i]
            if ct == "le":
                A_ub_rows.append(ai.copy()); b_ub_rows.append(bi)
            elif ct == "ge":
                A_ub_rows.append(-ai);       b_ub_rows.append(-bi)
            else:
                A_eq_rows.append(ai.copy()); b_eq_rows.append(bi)

        A_ub = np.array(A_ub_rows, dtype=float) if A_ub_rows else None
        b_ub = np.array(b_ub_rows, dtype=float) if b_ub_rows else None
        A_eq = np.array(A_eq_rows, dtype=float) if A_eq_rows else None
        b_eq = np.array(b_eq_rows, dtype=float) if A_eq_rows else None
        c_lp = -c if obj_type == "max" else c.copy()

        s3 = []
        s3.append("MAX &rArr; negate objective: MIN &minus;Z = &minus;c<sup>T</sup>x<br>"
                  if obj_type == "max" else
                  "MIN Z = c<sup>T</sup>x (no change needed)<br>")
        s3.append("Constraint conversion to linprog A_ub / A_eq form:<br>")
        s3.append("&nbsp;&nbsp;&bull; &le; : pass directly &nbsp; Ax &le; b<br>")
        s3.append("&nbsp;&nbsp;&bull; &ge; : multiply row by &minus;1 &nbsp; &minus;Ax &le; &minus;b<br>")
        s3.append("&nbsp;&nbsp;&bull; = &nbsp;: pass as equality &nbsp; A_eq x = b_eq")
        steps.append({"title": "Step 3: Convert to Standard LP Form", "body": "".join(s3)})

        # ── STEP 4: Solve primal ───────────────────────────────────────
        primal_res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                              bounds=[(0, None)]*n_vars, method="highs")
        primal_obj = None
        if primal_res.success:
            primal_obj = float(-primal_res.fun) if obj_type == "max" else float(primal_res.fun)

        steps.append({"title": "Step 4: Primal Optimal Solution",
                       "body": build_primal_text(primal_res, c, primal_obj, obj_type, n_vars)})

        # ── STEP 5: Dual formulation ───────────────────────────────────
        dual_c_vec = b_std          # n_con coefficients
        dual_A_mat = A_std.T        # n_vars x n_con
        dual_b_vec = c              # n_vars RHS

        steps.append({"title": "Step 5: Dual Problem Formulation",
                       "body": build_dual_text(dual_c_vec, dual_A_mat, dual_b_vec,
                                               obj_type, n_vars, n_con, ct_std)})

        # ── STEP 6: Solve dual ─────────────────────────────────────────
        dual_A_ub_rows, dual_b_ub_rows = [], []
        dual_bounds = []

        if obj_type == "max":
            # Dual MIN b.y  s.t.  A^T y >= c  =>  -A^T y <= -c
            for j in range(n_vars):
                dual_A_ub_rows.append(-A_std[:, j])
                dual_b_ub_rows.append(-c[j])
            for i in range(n_con):
                ct = ct_std[i]
                dual_bounds.append((0, None) if ct == "le" else
                                   (None, 0) if ct == "ge" else (None, None))
            dual_c_lp = dual_c_vec.copy()
        else:
            # Dual MAX b.y  s.t.  A^T y <= c  =>  linprog MIN -b.y s.t. A^T y <= c
            for j in range(n_vars):
                dual_A_ub_rows.append(A_std[:, j].copy())
                dual_b_ub_rows.append(c[j])
            for i in range(n_con):
                ct = ct_std[i]
                dual_bounds.append((0, None) if ct == "ge" else
                                   (None, 0) if ct == "le" else (None, None))
            dual_c_lp = -dual_c_vec

        dA_ub = np.array(dual_A_ub_rows, dtype=float) if dual_A_ub_rows else None
        db_ub = np.array(dual_b_ub_rows, dtype=float) if dual_b_ub_rows else None

        dual_res = linprog(dual_c_lp, A_ub=dA_ub, b_ub=db_ub,
                            bounds=dual_bounds, method="highs")
        dual_obj = None
        if dual_res.success:
            dual_obj = float(dual_res.fun) if obj_type == "max" else float(-dual_res.fun)

        steps.append({"title": "Step 6: Dual Optimal Solution",
                       "body": build_dual_sol_text(dual_res, dual_c_vec, dual_obj, obj_type, n_con)})

        # ── STEP 7: Strong duality ─────────────────────────────────────
        strong_duality = False
        cs_rows = []
        if primal_res.success and dual_res.success and primal_obj is not None and dual_obj is not None:
            strong_duality = bool(abs(primal_obj - dual_obj) < 1e-4)
            steps.append({"title": "Step 7: Strong Duality Theorem",
                           "body": build_strong_duality(primal_obj, dual_obj, strong_duality, obj_type)})

            # ── STEP 8: Complementary slackness ───────────────────────
            x = primal_res.x; y = dual_res.x
            cs_body = ["<b>Conditions:</b><br>"
                       "1. y<sub>i</sub> &bull; slack<sub>i</sub> = 0<br>"
                       "2. x<sub>j</sub> &bull; (c<sub>j</sub> &minus; A<sup>T</sup><sub>j</sub>y) = 0<br><br>"]
            for i in range(n_con):
                lhs_val = float(A[i] @ x)
                bi_orig = float(b[i])
                ct      = con_types[i]
                if ct == "le":
                    slack = bi_orig - lhs_val
                elif ct == "ge":
                    slack = lhs_val - bi_orig
                else:
                    slack = abs(lhs_val - bi_orig)
                yi   = float(y[i])
                prod = slack * yi
                ok   = bool(abs(prod) < 1e-4)
                mark = "&#10003;" if ok else "&#10007;"
                col  = "#4ade80" if ok else "#f87171"
                cs_body.append(
                    f'C{i+1} ({sign_html(ct)}): &nbsp;'
                    f'y<sub>{i+1}</sub>={fmt(yi)}, slack={fmt(slack)}, '
                    f'product={fmt(prod)} '
                    f'<span style="color:{col};font-size:16px">{mark}</span><br>')
                cs_rows.append({"constraint": i+1, "type": ct,
                                  "primal_slack": round(slack, 6),
                                  "dual_var": round(yi, 6), "satisfied": ok})
            steps.append({"title": "Step 8: Complementary Slackness",
                           "body": "".join(cs_body)})

        # ── Graph (2-variable only) ────────────────────────────────────
        graph_b64 = None
        if n_vars == 2 and primal_res.success:
            graph_b64 = build_graph(c, A, b, primal_res.x, obj_type, con_types)

        return jsonify({
            "primal": {
                "success": bool(primal_res.success),
                "status": primal_res.message,
                "optimal_value": round(primal_obj, 6) if primal_obj is not None else None,
                "variables": [round(float(v), 6) for v in primal_res.x] if primal_res.success else []
            },
            "dual": {
                "success": bool(dual_res.success),
                "status": dual_res.message,
                "optimal_value": round(dual_obj, 6) if dual_obj is not None else None,
                "variables": [round(float(v), 6) for v in dual_res.x] if dual_res.success else []
            },
            "strong_duality": strong_duality,
            "complementary_slackness": cs_rows,
            "steps":    steps,
            "graph":    graph_b64,
            "obj_type": obj_type,
            "n_vars":   n_vars
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400


# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt(v):
    v = float(v)
    if v == int(v): return str(int(v))
    return f"{v:.4g}"

def xsub(j, p="x"): return f"{p}<sub>{j+1}</sub>"

def sign_html(ct):
    return {"le": "&le;", "ge": "&ge;", "eq": "="}[ct]

def sign_plain(ct):
    return {"le": "≤", "ge": "≥", "eq": "="}[ct]


# ── Text builders ────────────────────────────────────────────────────────────

def build_problem_text(c, A, b, obj_type, n_vars, n_con, con_types):
    sense = "Maximize" if obj_type == "max" else "Minimize"
    obj   = " + ".join(f"{fmt(c[j])}{xsub(j)}" for j in range(n_vars))
    lines = [f"<b>{sense} Z = {obj}</b>", "<b>Subject to:</b>"]
    for i in range(n_con):
        row = " + ".join(f"{fmt(A[i][j])}{xsub(j)}" for j in range(n_vars))
        lines.append(f"&nbsp;&nbsp;{row} {sign_html(con_types[i])} {fmt(b[i])}")
    lines.append("&nbsp;&nbsp;x<sub>j</sub> &ge; 0")
    return "<br>".join(lines)


def build_primal_text(res, c, obj_val, obj_type, n_vars):
    if not res.success:
        return f"<b>Status:</b> {res.message}<br>The primal problem has no finite optimal solution."
    x = res.x
    sense = "Maximum" if obj_type == "max" else "Minimum"
    parts = [f"{fmt(c[j])}({fmt(x[j])})" for j in range(n_vars)]
    lines = [
        "<b>Optimal variables:</b> " + ", ".join(f"{xsub(j)} = {fmt(x[j])}" for j in range(n_vars)),
        f"<b>{sense} Z</b> = " + " + ".join(parts) + f" = <b>{fmt(obj_val)}</b>",
        "",
        "<i>Verification — check each constraint:</i>"
    ]
    return "<br>".join(lines)


def build_dual_text(dc, dA, db, obj_type, n_vars, n_con, ct_std):
    if obj_type == "max":
        sense    = "Minimize"; con_sign = "&ge;"
        note = ("Rule: Primal MAX &rarr; Dual MIN<br>"
                "&le; primal constraint &rarr; y<sub>i</sub> &ge; 0<br>"
                "&ge; primal constraint &rarr; y<sub>i</sub> &le; 0<br>"
                "= primal constraint &rarr; y<sub>i</sub> unrestricted")
    else:
        sense    = "Maximize"; con_sign = "&le;"
        note = ("Rule: Primal MIN &rarr; Dual MAX<br>"
                "&ge; primal constraint &rarr; y<sub>i</sub> &ge; 0<br>"
                "&le; primal constraint &rarr; y<sub>i</sub> &le; 0<br>"
                "= primal constraint &rarr; y<sub>i</sub> unrestricted")
    obj   = " + ".join(f"{fmt(dc[i])}{xsub(i,'y')}" for i in range(n_con))
    lines = [f"<b>{sense} W = {obj}</b>", "<b>Subject to:</b>"]
    for j in range(n_vars):
        row = " + ".join(f"{fmt(dA[j][i])}{xsub(i,'y')}" for i in range(n_con))
        lines.append(f"&nbsp;&nbsp;{row} {con_sign} {fmt(db[j])}")
    # Dual variable sign restrictions
    y_signs = []
    for i in range(n_con):
        ct = ct_std[i]
        if obj_type == "max":
            r = "&ge; 0" if ct == "le" else ("&le; 0" if ct == "ge" else "unrestricted")
        else:
            r = "&ge; 0" if ct == "ge" else ("&le; 0" if ct == "le" else "unrestricted")
        y_signs.append(f"{xsub(i,'y')} {r}")
    lines.append("&nbsp;&nbsp;" + ",&nbsp; ".join(y_signs) + "<br>")
    lines.append(f"<i>{note}</i>")
    return "<br>".join(lines)


def build_dual_sol_text(res, dc, obj_val, obj_type, n_con):
    if not res.success:
        return f"<b>Status:</b> {res.message}"
    y = res.x
    sense = "Minimum" if obj_type == "max" else "Maximum"
    parts = [f"{fmt(dc[i])}({fmt(y[i])})" for i in range(n_con)]
    return ("<b>Optimal dual variables:</b> " +
            ", ".join(f"{xsub(i,'y')} = {fmt(y[i])}" for i in range(n_con)) +
            f"<br><b>{sense} W</b> = " + " + ".join(parts) + f" = <b>{fmt(obj_val)}</b>")


def build_strong_duality(p_val, d_val, holds, obj_type):
    gap = abs(p_val - d_val)
    if holds:
        return (f"<b>&#10003; Strong Duality holds!</b><br><br>"
                f"Primal optimal Z* = <b>{fmt(p_val)}</b><br>"
                f"Dual optimal W* = <b>{fmt(d_val)}</b><br>"
                f"Gap = {gap:.2e} &approx; 0<br><br>"
                f"<i>Strong Duality Theorem: If the primal LP has an optimal solution, "
                f"so does the dual, and Z* = W*</i>")
    else:
        return (f"<b>&#10007; Duality gap detected</b><br>"
                f"Z* = {fmt(p_val)}, W* = {fmt(d_val)}, Gap = {gap:.4f}<br>"
                f"This may indicate numerical issues or mixed constraint types affecting classical duality.")


# ── Graph ────────────────────────────────────────────────────────────────────

def build_graph(c, A, b, x_opt, obj_type, con_types):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.spatial import ConvexHull

        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        fig.patch.set_facecolor("#0d0d1a")
        ax.set_facecolor("#111120")

        # Dynamic axis range
        x_max = max(float(x_opt[0]) * 1.5, 10.0)
        y_max = max(float(x_opt[1]) * 1.5, 10.0)
        for i in range(len(b)):
            if abs(A[i][0]) > 1e-9: x_max = max(x_max, abs(float(b[i])/A[i][0]) * 1.3)
            if abs(A[i][1]) > 1e-9: y_max = max(y_max, abs(float(b[i])/A[i][1]) * 1.3)
        x_max = min(x_max, 300); y_max = min(y_max, 300)
        x_vals = np.linspace(0, x_max, 800)

        palette = ["#7c6af7","#f7a26a","#4ade80","#f87171","#38bdf8","#e879f9"]

        for i in range(len(b)):
            a0, a1, bi = float(A[i][0]), float(A[i][1]), float(b[i])
            col = palette[i % len(palette)]
            lbl = f"C{i+1}: {fmt(a0)}x₁+{fmt(a1)}x₂ {sign_plain(con_types[i])} {fmt(bi)}"
            if abs(a1) > 1e-9:
                y_line = (bi - a0*x_vals) / a1
                y_line = np.where((y_line >= -1) & (y_line <= y_max*1.5), y_line, np.nan)
                ax.plot(x_vals, y_line, color=col, lw=2, label=lbl, zorder=4)
            elif abs(a0) > 1e-9:
                ax.axvline(bi/a0, color=col, lw=2, label=lbl, zorder=4)

        verts = feasible_vertices(A, b, x_max, y_max, con_types)
        if len(verts) >= 3:
            pts = np.array(verts)
            try:
                hull = ConvexHull(pts)
                poly = pts[hull.vertices]
                ax.fill(poly[:,0], poly[:,1], color="#7c6af7", alpha=0.12, zorder=1, label="Feasible region")
                for pt in poly:
                    ax.plot(pt[0], pt[1], "o", color="#7c6af7", markersize=5, alpha=0.6, zorder=5)
            except Exception:
                pass

        if abs(c[1]) > 1e-9:
            obj_val = c[0]*float(x_opt[0]) + c[1]*float(x_opt[1])
            y_obj   = (obj_val - c[0]*x_vals) / c[1]
            y_obj   = np.where((y_obj >= -1) & (y_obj <= y_max*1.5), y_obj, np.nan)
            ax.plot(x_vals, y_obj, "--", color="#fbbf24", lw=1.5, alpha=0.7,
                    label=f"Z={fmt(obj_val)} (objective)", zorder=3)

        norm = math.sqrt(float(c[0])**2 + float(c[1])**2)
        if norm > 1e-9:
            dx, dy = float(c[0])/norm, float(c[1])/norm
            scale  = min(x_max, y_max) * 0.14
            sd     = 1 if obj_type == "max" else -1
            ox, oy = float(x_opt[0]), float(x_opt[1])
            ax.annotate("", xy=(ox+sd*dx*scale, oy+sd*dy*scale), xytext=(ox, oy),
                        arrowprops=dict(arrowstyle="-|>", color="#fbbf24", lw=2), zorder=8)

        ax.scatter([x_opt[0]], [x_opt[1]], color="#fbbf24", s=200, zorder=10,
                   edgecolors="white", linewidths=1.5,
                   label=f"Optimal: ({fmt(x_opt[0])}, {fmt(x_opt[1])})")
        ax.annotate(f"  Z*={fmt(c[0]*float(x_opt[0])+c[1]*float(x_opt[1]))}",
                    (float(x_opt[0]), float(x_opt[1])),
                    color="#fbbf24", fontsize=9, zorder=11,
                    xytext=(6, 6), textcoords="offset points")

        ax.set_xlim(0, x_max); ax.set_ylim(0, y_max)
        ax.set_xlabel("x₁", color="#ccc", fontsize=12)
        ax.set_ylabel("x₂", color="#ccc", fontsize=12)
        ax.tick_params(colors="#666")
        for sp in ax.spines.values(): sp.set_edgecolor("#222244")
        ax.grid(True, color="#1e1e38", lw=0.6, linestyle="--")
        ax.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e",
                  edgecolor="#333", labelcolor="white", framealpha=0.92)
        ax.set_title(f"{'Maximisation' if obj_type=='max' else 'Minimisation'} — Graphical Method",
                     color="#e8e8f0", fontsize=12, pad=10)
        fig.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print("Graph error:", e)
        return None


def feasible_vertices(A, b, x_max, y_max, con_types):
    extra_A  = [[-1,0],[0,-1],[1,0],[0,1]]
    extra_b  = [0, 0, x_max, y_max]
    extra_ct = ["le","le","le","le"]
    rows = list(A)      + extra_A
    rhs  = list(b)      + extra_b
    cts  = list(con_types) + extra_ct
    pts  = []
    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            try:
                M = np.array([rows[i], rows[j]], dtype=float)
                if abs(np.linalg.det(M)) < 1e-10: continue
                pt = np.linalg.solve(M, [rhs[i], rhs[j]])
                if pt[0] < -1e-6 or pt[1] < -1e-6: continue
                if pt[0] > x_max*1.01 or pt[1] > y_max*1.01: continue
                feasible = True
                for k in range(len(A)):
                    lhs_k = A[k][0]*pt[0] + A[k][1]*pt[1]
                    ct = con_types[k]
                    if ct == "le" and lhs_k > b[k]+1e-6: feasible=False; break
                    if ct == "ge" and lhs_k < b[k]-1e-6: feasible=False; break
                    if ct == "eq" and abs(lhs_k - b[k]) > 1e-6: feasible=False; break
                if feasible:
                    pts.append(pt.tolist())
            except Exception:
                pass
    return pts

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
# if __name__ == "__main__":
#     app.run(debug=True, port=5050)
