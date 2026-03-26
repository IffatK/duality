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
        con_types = data.get("constraint_types",
                             ["le" if obj_type == "max" else "ge"] * len(b))
        n_vars = len(c)
        n_con  = len(b)
        steps  = []

        # ── STEP 1: Original Problem ───────────────────────────────────
        steps.append({"title": "Step 1: Original Problem",
                       "body": build_problem_text(c, A, b, obj_type, n_vars, n_con, con_types)})

        # ── STEP 2: Handle negative RHS ────────────────────────────────
        A_std = A.copy(); b_std = b.copy(); ct_std = list(con_types)
        flipped = []
        for i in range(n_con):
            if b_std[i] < 0:
                A_std[i] *= -1; b_std[i] *= -1
                if ct_std[i] == "le":   ct_std[i] = "ge"
                elif ct_std[i] == "ge": ct_std[i] = "le"
                flipped.append(i + 1)

        if flipped:
            details = []
            for idx in [x - 1 for x in flipped]:
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

        # ── STEP 3: Standard LP form ───────────────────────────────────
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

        # ── STEP 4: Solve Primal ───────────────────────────────────────
        primal_res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                              bounds=[(0, None)] * n_vars, method="highs")
        primal_obj = None
        if primal_res.success:
            primal_obj = float(-primal_res.fun) if obj_type == "max" else float(primal_res.fun)

        steps.append({"title": "Step 4: Primal Optimal Solution",
                       "body": build_primal_text(primal_res, c, primal_obj, obj_type, n_vars)})

        # ── BIG-M METHOD (if infeasible) ───────────────────────────────
        big_m_result = None
        big_m_steps  = []
        if not primal_res.success:
            big_m_result, big_m_steps = big_m_method(c, A_std, b_std, ct_std, obj_type, n_vars, n_con)
            steps.append({"title": "Step 4b: Big-M Method (Infeasibility Recovery)",
                           "body": big_m_steps[0] if big_m_steps else "Big-M analysis complete."})
            if len(big_m_steps) > 1:
                for idx, bms in enumerate(big_m_steps[1:], 1):
                    steps.append({"title": f"Step 4b-{idx}: Big-M — {bms['title']}",
                                   "body": bms["body"]})

        # ── Simplex Tableau (full manual step-by-step) ────────────────
        simplex_html = build_simplex_tableau(c, A_std, b_std, ct_std, obj_type, n_vars, n_con)
        steps.append({"title": "Step 4c: Simplex Method — Full Tableau",
                       "body": simplex_html})

        # ── Pandas Simulation Table ────────────────────────────────────
        sim_table_html = build_pandas_simulation(c, A, b, con_types, obj_type, n_vars, n_con,
                                                  primal_res, primal_obj)
        steps.append({"title": "Step 4d: Sensitivity Simulation (Pandas)",
                       "body": sim_table_html})

        # ── STEP 5: Dual Formulation ───────────────────────────────────
        dual_c_vec = b_std
        dual_A_mat = A_std.T
        dual_b_vec = c

        steps.append({"title": "Step 5: Dual Problem Formulation",
                       "body": build_dual_text(dual_c_vec, dual_A_mat, dual_b_vec,
                                               obj_type, n_vars, n_con, ct_std)})

        # ── STEP 6: Solve Dual ─────────────────────────────────────────
        dual_A_ub_rows, dual_b_ub_rows = [], []
        dual_bounds = []

        if obj_type == "max":
            for j in range(n_vars):
                dual_A_ub_rows.append(-A_std[:, j])
                dual_b_ub_rows.append(-c[j])
            for i in range(n_con):
                ct = ct_std[i]
                dual_bounds.append((0, None) if ct == "le" else
                                   (None, 0) if ct == "ge" else (None, None))
            dual_c_lp = dual_c_vec.copy()
        else:
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

        # ── STEP 7: Strong Duality ─────────────────────────────────────
        strong_duality = False
        cs_rows = []
        if primal_res.success and dual_res.success and primal_obj is not None and dual_obj is not None:
            strong_duality = bool(abs(primal_obj - dual_obj) < 1e-4)
            steps.append({"title": "Step 7: Strong Duality Theorem",
                           "body": build_strong_duality(primal_obj, dual_obj, strong_duality, obj_type)})

            # ── STEP 8: Complementary Slackness ───────────────────────
            x = primal_res.x; y = dual_res.x
            cs_body = ["<b>Conditions:</b><br>"
                       "1. y<sub>i</sub> &bull; slack<sub>i</sub> = 0<br>"
                       "2. x<sub>j</sub> &bull; (c<sub>j</sub> &minus; A<sup>T</sup><sub>j</sub>y) = 0<br><br>"]
            for i in range(n_con):
                lhs_val = float(A[i] @ x)
                bi_orig = float(b[i])
                ct      = con_types[i]
                if ct == "le":   slack = bi_orig - lhs_val
                elif ct == "ge": slack = lhs_val - bi_orig
                else:            slack = abs(lhs_val - bi_orig)
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

        # ── Graph (2-variable primal, works for 2–5 constraints) ──────
        graph_b64 = None
        if n_vars == 2 and (primal_res.success or (big_m_result and big_m_result.get("success"))):
            x_opt = primal_res.x if primal_res.success else np.array(big_m_result["x"])
            graph_b64 = build_graph(c, A, b, x_opt, obj_type, con_types, n_con)

        # ── Big-M graph if primal infeasible ──────────────────────────
        big_m_graph_b64 = None
        if not primal_res.success and n_vars == 2 and big_m_result and big_m_result.get("success"):
            big_m_graph_b64 = build_big_m_graph(c, A, b, np.array(big_m_result["x"]),
                                                  obj_type, con_types, n_con)

        return jsonify({
            "primal": {
                "success": bool(primal_res.success),
                "status":  primal_res.message,
                "optimal_value": round(primal_obj, 6) if primal_obj is not None else None,
                "variables": [round(float(v), 6) for v in primal_res.x] if primal_res.success else []
            },
            "dual": {
                "success": bool(dual_res.success),
                "status":  dual_res.message,
                "optimal_value": round(dual_obj, 6) if dual_obj is not None else None,
                "variables": [round(float(v), 6) for v in dual_res.x] if dual_res.success else []
            },
            "big_m": big_m_result,
            "strong_duality": strong_duality,
            "complementary_slackness": cs_rows,
            "steps":          steps,
            "graph":          graph_b64,
            "big_m_graph":    big_m_graph_b64,
            "obj_type":       obj_type,
            "n_vars":         n_vars
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400


# ── Big-M Method ─────────────────────────────────────────────────────────────

def big_m_method(c, A_std, b_std, ct_std, obj_type, n_vars, n_con):
    """
    Implement Big-M method to handle infeasible LP.
    Adds artificial variables with large penalty M.
    Returns (result_dict, steps_list).
    """
    M = 1e6  # Big-M penalty
    steps = []

    # Build augmented problem with slack/surplus/artificial variables
    # For each constraint:
    #   le: add slack s_i >= 0   (no artificial needed)
    #   ge: subtract surplus s_i >= 0, add artificial a_i >= 0
    #   eq: add artificial a_i >= 0

    n_slack      = sum(1 for ct in ct_std if ct == "le")
    n_surplus    = sum(1 for ct in ct_std if ct == "ge")
    n_artificial = sum(1 for ct in ct_std if ct in ("ge", "eq"))

    total_vars = n_vars + n_slack + n_surplus + n_artificial

    # Objective: penalize artificial variables
    if obj_type == "max":
        c_aug = np.concatenate([c, np.zeros(n_slack + n_surplus), -M * np.ones(n_artificial)])
        c_lp  = -c_aug  # linprog minimizes
    else:
        c_aug = np.concatenate([c, np.zeros(n_slack + n_surplus), M * np.ones(n_artificial)])
        c_lp  = c_aug

    # Build equality constraints (all constraints become equality with slacks)
    A_eq_aug = np.zeros((n_con, total_vars))
    b_eq_aug = b_std.copy()

    slack_idx = n_vars
    surplus_idx = n_vars + n_slack
    art_idx = n_vars + n_slack + n_surplus

    art_col = art_idx
    for i, ct in enumerate(ct_std):
        A_eq_aug[i, :n_vars] = A_std[i]
        if ct == "le":
            A_eq_aug[i, slack_idx] = 1
            slack_idx += 1
        elif ct == "ge":
            A_eq_aug[i, surplus_idx] = -1
            surplus_idx += 1
            A_eq_aug[i, art_col] = 1
            art_col += 1
        else:  # eq
            A_eq_aug[i, art_col] = 1
            art_col += 1

    bounds = [(0, None)] * total_vars
    res = linprog(c_lp, A_eq=A_eq_aug, b_eq=b_eq_aug, bounds=bounds, method="highs")

    # Check if artificial variables are zero at optimum (true feasibility)
    truly_feasible = False
    art_vals = []
    if res.success:
        art_vals = res.x[n_vars + n_slack + n_surplus:]
        truly_feasible = bool(np.all(np.abs(art_vals) < 1e-6))

    # Reconstruct original objective value
    big_m_obj = None
    big_m_x   = None
    if res.success and truly_feasible:
        big_m_x = res.x[:n_vars].tolist()
        raw_obj = float(c @ res.x[:n_vars])
        big_m_obj = raw_obj if obj_type == "min" else raw_obj

    # Build explanation
    var_names = ([f"x<sub>{j+1}</sub>" for j in range(n_vars)] +
                 [f"s<sub>{k+1}</sub>" for k in range(n_slack)] +
                 [f"e<sub>{k+1}</sub>" for k in range(n_surplus)] +
                 [f"a<sub>{k+1}</sub>" for k in range(n_artificial)])

    obj_terms = []
    for j in range(n_vars):
        obj_terms.append(f"{fmt(c_aug[j])}{var_names[j]}")
    for k in range(n_slack + n_surplus):
        obj_terms.append(f"0&middot;{var_names[n_vars + k]}")
    for k in range(n_artificial):
        sign = "&minus;M" if obj_type == "max" else "+M"
        obj_terms.append(f"{sign}&middot;{var_names[n_vars + n_slack + n_surplus + k]}")

    body = (f"<b>Big-M Method</b>: Penalize artificial variables with M = {int(M):,}<br><br>"
            f"<b>Augmented objective:</b><br>"
            f"&nbsp;&nbsp;{'Max' if obj_type=='max' else 'Min'} Z = {' + '.join(obj_terms)}<br><br>"
            f"<b>Augmented constraints</b> (all converted to equality):<br>")

    for i, ct in enumerate(ct_std):
        row_terms = " + ".join(f"{fmt(A_eq_aug[i][k])}{var_names[k]}"
                                for k in range(total_vars) if abs(A_eq_aug[i][k]) > 1e-12)
        body += f"&nbsp;&nbsp;{row_terms} = {fmt(b_eq_aug[i])}<br>"

    body += f"<br><b>All variables &ge; 0</b><br><br>"

    if res.success:
        if truly_feasible:
            body += (f'<span style="color:#4ade80"><b>&#10003; Big-M Solution Found</b></span><br>'
                     f"Artificial variables are zero — problem IS feasible under relaxed objective.<br>"
                     f"<b>x* = ({', '.join(fmt(v) for v in big_m_x)})</b><br>"
                     f"<b>Original Z* = {fmt(big_m_obj)}</b>")
        else:
            body += (f'<span style="color:#f87171"><b>&#10007; Truly Infeasible</b></span><br>'
                     f"Artificial variables remain positive at optimum:<br>"
                     + "".join(f"&nbsp;&nbsp;a<sub>{k+1}</sub> = {fmt(art_vals[k])}<br>"
                                for k in range(n_artificial))
                     + "The original LP has no feasible solution.")
    else:
        body += (f'<span style="color:#f87171"><b>&#10007; Big-M also infeasible / unbounded</b></span><br>'
                 f"Status: {res.message}")

    step_list = [body]

    # Tableau-style pandas table
    tableau_html = build_big_m_tableau(A_eq_aug, b_eq_aug, c_aug, var_names, n_con, total_vars, M, obj_type)
    step_list.append({"title": "Augmented Tableau", "body": tableau_html})

    result = {
        "success": bool(res.success and truly_feasible),
        "x":       big_m_x or [],
        "obj":     big_m_obj,
        "art_zero": truly_feasible
    }
    return result, step_list


def build_big_m_tableau(A_eq, b_eq, c_aug, var_names, n_con, total_vars, M, obj_type):
    """Build an HTML table showing the Big-M augmented tableau."""
    header = "".join(f"<th>{v}</th>" for v in var_names) + "<th>RHS</th>"
    rows = ""
    for i in range(n_con):
        cells = "".join(f"<td>{fmt(A_eq[i][j])}</td>" for j in range(total_vars))
        cells += f"<td style='color:var(--teal);font-weight:600'>{fmt(b_eq[i])}</td>"
        rows += f"<tr><td style='color:var(--muted)'>R{i+1}</td>{cells}</tr>"

    obj_sense = "MAX (negated)" if obj_type == "max" else "MIN"
    c_row = "".join(f"<td style='color:var(--indigo)'>{fmt(v)}</td>" for v in c_aug)
    c_row += "<td>—</td>"

    return (f"<b>Augmented Tableau ({obj_sense}):</b><br><br>"
            f'<div style="overflow-x:auto">'
            f'<table style="border-collapse:collapse;font-family:var(--mono);font-size:11px;width:100%">'
            f'<thead><tr><th style="color:var(--muted)"></th>{header}</tr></thead>'
            f'<tbody>{rows}'
            f'<tr style="border-top:2px solid var(--border2)">'
            f'<td style="color:var(--amber)">c</td>{c_row}</tr>'
            f'</tbody></table></div>'
            f'<br><i style="color:var(--muted);font-size:11px">M = {int(M):,} (penalty for artificial variables)</i>')


# ── Full Simplex Tableau (manual step-by-step) ───────────────────────────────

def build_simplex_tableau(c, A, b, con_types, obj_type, n_vars, n_con):
    """
    Perform the Simplex method manually and render each iteration as an HTML table.
    Columns: Iteration | Basic Var | x1..xn | s1/e1..sm | b_i (RHS) | Ratio
    
    Slack/surplus variable logic:
      MAX problem:
        <=  → add slack s_i (+1)          → s_i starts in basis
        >=  → subtract surplus e_i (-1), add artificial a_i (+1) → a_i starts in basis
        =   → add artificial a_i (+1)     → a_i starts in basis
      MIN problem:
        >=  → subtract surplus e_i (-1), add artificial a_i (+1) → a_i starts in basis
        <=  → add slack s_i (+1)          → s_i starts in basis
        =   → add artificial a_i (+1)     → a_i starts in basis
    """
    MAX_ITERS = 20
    M_BIG = 1e6  # Big-M for artificial variables when needed

    # ── 1. Build augmented tableau ──────────────────────────────────────────
    # Determine which extra variables to add
    var_labels = [f"x<sub>{j+1}</sub>" for j in range(n_vars)]
    slack_map   = {}   # constraint index → column index (slack s_i)
    surplus_map = {}   # constraint index → column index (surplus e_i)
    art_map     = {}   # constraint index → column index (artificial a_i)

    extra_cols = []   # list of (label, col_type)
    col_idx = n_vars
    for i, ct in enumerate(con_types):
        if ct == "le":
            slack_map[i] = col_idx
            extra_cols.append((f"s<sub>{i+1}</sub>", "slack"))
            col_idx += 1
        elif ct == "ge":
            surplus_map[i] = col_idx
            extra_cols.append((f"e<sub>{i+1}</sub>", "surplus"))
            col_idx += 1
            art_map[i] = col_idx
            extra_cols.append((f"a<sub>{i+1}</sub>", "art"))
            col_idx += 1
        else:  # eq
            art_map[i] = col_idx
            extra_cols.append((f"a<sub>{i+1}</sub>", "art"))
            col_idx += 1

    total_cols = col_idx
    all_labels = var_labels + [lbl for lbl, _ in extra_cols]

    # Build constraint matrix (augmented)
    T = np.zeros((n_con, total_cols + 1))  # last col = RHS
    for i in range(n_con):
        for j in range(n_vars):
            T[i, j] = A[i, j]
        if i in slack_map:
            T[i, slack_map[i]] = 1.0
        if i in surplus_map:
            T[i, surplus_map[i]] = -1.0
        if i in art_map:
            T[i, art_map[i]] = 1.0
        T[i, -1] = b[i]

    # Build objective row (cj row) for simplex
    # For MAX: objective is c (we maximize cTx)
    # For MIN: objective is -c (we negate to use same maximization simplex)
    if obj_type == "max":
        c_obj = np.zeros(total_cols)
        for j in range(n_vars): c_obj[j] = float(c[j])
        for i in art_map.values(): c_obj[i] = -M_BIG  # penalize
    else:
        c_obj = np.zeros(total_cols)
        for j in range(n_vars): c_obj[j] = -float(c[j])  # negate for min→max
        for i in art_map.values(): c_obj[i] = -M_BIG

    # Initial basis: slack or artificial variables
    basis = []
    for i in range(n_con):
        if i in art_map:
            basis.append(art_map[i])
        else:
            basis.append(slack_map[i])

    # ── 2. Simplex iterations ───────────────────────────────────────────────
    has_artificials = len(art_map) > 0
    iterations = []

    def compute_zj_cjzj(T, basis, c_obj, n_con, total_cols):
        """Compute Z row (cB * column) and cj - Zj."""
        cb = np.array([c_obj[b] for b in basis])
        zj = np.array([float(cb @ T[:, j]) for j in range(total_cols)])
        cj_zj = c_obj - zj
        z_val = float(cb @ T[:, -1])
        return zj, cj_zj, z_val

    def record_iteration(T, basis, c_obj, it_num, pivot_col=None, pivot_row=None,
                          entering=None, leaving=None, note=""):
        n_rows, _ = T.shape
        ratios = []
        if pivot_col is not None:
            for i in range(n_rows):
                col_val = T[i, pivot_col]
                rhs_val = T[i, -1]
                if col_val > 1e-9:
                    ratios.append(round(float(rhs_val / col_val), 4))
                else:
                    ratios.append(None)  # no ratio (non-positive)

        zj, cj_zj, z_val = compute_zj_cjzj(T, basis, c_obj, n_rows, total_cols)

        rows_data = []
        for i in range(n_rows):
            bv = basis[i]
            bv_label = all_labels[bv] if bv < len(all_labels) else f"v{bv}"
            row_vals = [round(float(T[i, j]), 4) for j in range(total_cols)]
            rhs = round(float(T[i, -1]), 4)
            ratio = ratios[i] if ratios else None
            rows_data.append({
                "basic_var": bv_label,
                "cb": round(float(c_obj[bv]), 4),
                "vals": row_vals,
                "rhs": rhs,
                "ratio": ratio,
                "is_pivot_row": (pivot_row == i)
            })

        iterations.append({
            "iter": it_num,
            "rows": rows_data,
            "zj": [round(float(v), 4) for v in zj],
            "cj_zj": [round(float(v), 4) for v in cj_zj],
            "z_val": round(z_val, 4),
            "pivot_col": pivot_col,
            "pivot_row": pivot_row,
            "entering": entering,
            "leaving": leaving,
            "note": note,
            "obj_type": obj_type
        })

    # ── Main simplex loop ───────────────────────────────────────────────────
    for it in range(MAX_ITERS + 1):
        zj, cj_zj, z_val = compute_zj_cjzj(T, basis, c_obj, n_con, total_cols)

        # Find entering variable: most positive cj - zj
        # Exclude artificial variables from entering if they've left basis
        max_cj_zj = -1e-9
        enter_col = -1
        for j in range(total_cols):
            if cj_zj[j] > max_cj_zj:
                max_cj_zj = cj_zj[j]
                enter_col = j

        if enter_col == -1 or max_cj_zj <= 1e-9:
            # Optimal
            record_iteration(T, basis, c_obj, it + 1, note="✓ Optimal — all c<sub>j</sub>−Z<sub>j</sub> ≤ 0")
            break

        # Find leaving variable: minimum ratio test
        min_ratio = float("inf")
        leave_row = -1
        for i in range(n_con):
            if T[i, enter_col] > 1e-9:
                ratio = T[i, -1] / T[i, enter_col]
                if ratio < min_ratio - 1e-9:
                    min_ratio = ratio
                    leave_row = i
                elif abs(ratio - min_ratio) < 1e-9 and i < leave_row:
                    leave_row = i  # Bland's tie-breaking

        if leave_row == -1:
            record_iteration(T, basis, c_obj, it + 1, note="⚠ Unbounded — no positive pivot element")
            break

        entering_lbl = all_labels[enter_col] if enter_col < len(all_labels) else f"v{enter_col}"
        leaving_lbl  = all_labels[basis[leave_row]] if basis[leave_row] < len(all_labels) else f"v{basis[leave_row]}"

        record_iteration(T, basis, c_obj, it + 1,
                         pivot_col=enter_col, pivot_row=leave_row,
                         entering=entering_lbl, leaving=leaving_lbl,
                         note=f"Entering: {entering_lbl} &nbsp;|&nbsp; Leaving: {leaving_lbl} &nbsp;|&nbsp; Pivot = {round(float(T[leave_row, enter_col]), 4)}")

        # Pivot
        pivot_val = T[leave_row, enter_col]
        T[leave_row, :] /= pivot_val
        for i in range(n_con):
            if i != leave_row:
                T[i, :] -= T[i, enter_col] * T[leave_row, :]
        basis[leave_row] = enter_col

        if it == MAX_ITERS:
            record_iteration(T, basis, c_obj, it + 2, note="⚠ Max iterations reached")

    # ── 3. Render HTML ──────────────────────────────────────────────────────
    def render_val(v):
        if v is None: return "—"
        if isinstance(v, float):
            if abs(v) > 9e5: return "M" if v > 0 else "−M"
            if v == int(v): return str(int(v))
            return f"{v:.3g}"
        return str(v)

    # Build explanation of variables used
    var_explain = []
    for i, ct in enumerate(con_types):
        cname = f"Constraint {i+1} ({sign_plain(ct)})"
        if ct == "le":
            var_explain.append(f"{cname}: add <b>slack s<sub>{i+1}</sub></b> (+1) &rarr; s<sub>{i+1}</sub> in initial basis")
        elif ct == "ge":
            var_explain.append(f"{cname}: subtract <b>surplus e<sub>{i+1}</sub></b> (−1) + add <b>artificial a<sub>{i+1}</sub></b> (+1) &rarr; a<sub>{i+1}</sub> in initial basis")
        else:
            var_explain.append(f"{cname}: add <b>artificial a<sub>{i+1}</sub></b> (+1) &rarr; a<sub>{i+1}</sub> in initial basis")

    sense_note = ("Objective: <b>MAX</b> Z = c<sup>T</sup>x &nbsp;→&nbsp; Maximize rows use <i>c<sub>j</sub> − Z<sub>j</sub></i>. "
                  "Enter variable with <b>most positive</b> c<sub>j</sub>−Z<sub>j</sub>."
                  if obj_type == "max" else
                  "Objective: <b>MIN</b> Z &nbsp;→&nbsp; Converted to MAX (−Z). "
                  "Enter variable with <b>most positive</b> c<sub>j</sub>−Z<sub>j</sub> of negated problem.")

    html = (f'<div style="margin-bottom:14px">'
            f'<div style="font-size:12px;color:var(--teal);font-weight:600;margin-bottom:8px">Variable Introduction</div>'
            f'<div style="font-size:11.5px;color:#94a3b8;line-height:2">' +
            "<br>".join(var_explain) +
            f'</div><br>'
            f'<div style="font-size:11.5px;color:var(--amber)">{sense_note}</div>'
            f'</div>')

    # Render each iteration
    for it_data in iterations:
        it_num   = it_data["iter"]
        rows     = it_data["rows"]
        zj       = it_data["zj"]
        cj_zj    = it_data["cj_zj"]
        z_val    = it_data["z_val"]
        p_col    = it_data["pivot_col"]
        note     = it_data["note"]

        # cj header row values
        cj_vals  = [render_val(float(c_obj[j])) for j in range(total_cols)]

        is_optimal = "Optimal" in note or "optimal" in note

        border_col = "#4ade80" if is_optimal else ("#f87171" if "Unbounded" in note or "Max iter" in note else "#6366f1")

        html += (f'<div style="margin-bottom:22px;border-left:3px solid {border_col};padding-left:12px">'
                 f'<div style="font-family:var(--mono);font-size:12px;color:{border_col};font-weight:700;margin-bottom:6px">'
                 f'Iteration {it_num} &nbsp;<span style="font-weight:400;color:#94a3b8;font-size:11px">{note}</span></div>')

        html += '<div style="overflow-x:auto"><table style="border-collapse:collapse;font-family:var(--mono);font-size:11px;width:100%;min-width:520px">'

        # --- cj header row ---
        th_style  = "padding:5px 8px;border:1px solid var(--border2);text-align:center;font-size:10px"
        html += f'<thead>'
        html += f'<tr style="background:rgba(99,102,241,.06)">'
        html += f'<th style="{th_style};color:var(--muted)">Iter</th>'
        html += f'<th style="{th_style};color:var(--muted)">c<sub>B</sub></th>'
        html += f'<th style="{th_style};color:var(--muted)">Basic Var</th>'
        # cj header row
        for j, lbl in enumerate(all_labels):
            hi = "color:var(--amber)" if j == p_col else "color:var(--indigo)"
            html += f'<th style="{th_style};{hi}">{lbl}<br><small style="color:var(--muted);font-size:9px">c<sub>j</sub>={render_val(float(c_obj[j]))}</small></th>'
        html += f'<th style="{th_style};color:var(--teal)">b<sub>i</sub> (RHS)</th>'
        html += f'<th style="{th_style};color:var(--amber)">Ratio</th>'
        html += '</tr></thead><tbody>'

        # --- data rows ---
        for r in rows:
            pivot_hi = "background:rgba(99,102,241,.1)" if r["is_pivot_row"] else ""
            html += f'<tr style="{pivot_hi}">'
            html += f'<td style="padding:5px 8px;border:1px solid var(--border2);text-align:center;color:var(--muted)">{it_num}</td>'
            html += f'<td style="padding:5px 8px;border:1px solid var(--border2);text-align:center;color:var(--indigo)">{render_val(r["cb"])}</td>'
            html += f'<td style="padding:5px 8px;border:1px solid var(--border2);color:var(--teal);font-weight:600;text-align:center">{r["basic_var"]}</td>'
            for j, v in enumerate(r["vals"]):
                cell_hi = "background:rgba(251,191,36,.1);color:var(--amber);font-weight:600" if j == p_col else "color:var(--text)"
                if r["is_pivot_row"] and j == p_col:
                    cell_hi = "background:rgba(99,102,241,.25);color:#fff;font-weight:700"
                html += f'<td style="padding:5px 8px;border:1px solid var(--border2);text-align:center;{cell_hi}">{render_val(v)}</td>'
            html += f'<td style="padding:5px 8px;border:1px solid var(--border2);text-align:center;color:var(--teal);font-weight:600">{render_val(r["rhs"])}</td>'
            # Ratio
            ratio_str = render_val(r["ratio"]) if r["ratio"] is not None else "—"
            ratio_hi  = "color:var(--amber);font-weight:700" if r["is_pivot_row"] and r["ratio"] is not None else "color:var(--muted)"
            if r["is_pivot_row"] and r["ratio"] is not None:
                ratio_str += " ←min"
            html += f'<td style="padding:5px 8px;border:1px solid var(--border2);text-align:center;{ratio_hi}">{ratio_str}</td>'
            html += '</tr>'

        # --- Zj row ---
        html += '<tr style="border-top:2px solid var(--border2);background:rgba(20,184,166,.04)">'
        html += f'<td colspan="3" style="padding:5px 8px;border:1px solid var(--border2);color:var(--teal);font-weight:700;text-align:right">Z<sub>j</sub></td>'
        for j, v in enumerate(zj):
            html += f'<td style="padding:5px 8px;border:1px solid var(--border2);text-align:center;color:var(--teal)">{render_val(v)}</td>'
        html += f'<td style="padding:5px 8px;border:1px solid var(--border2);text-align:center;color:var(--teal);font-weight:700">{render_val(z_val)}</td>'
        html += '<td style="padding:5px 8px;border:1px solid var(--border2)"></td>'
        html += '</tr>'

        # --- cj - Zj row ---
        html += '<tr style="background:rgba(99,102,241,.04)">'
        html += f'<td colspan="3" style="padding:5px 8px;border:1px solid var(--border2);color:var(--indigo);font-weight:700;text-align:right">c<sub>j</sub>−Z<sub>j</sub></td>'
        for j, v in enumerate(cj_zj):
            # Highlight the entering column
            is_enter = (p_col == j)
            val_color = "color:var(--amber);font-weight:700" if is_enter else ("color:var(--green)" if v > 1e-9 else ("color:var(--red)" if v < -1e-9 else "color:var(--muted)"))
            html += f'<td style="padding:5px 8px;border:1px solid var(--border2);text-align:center;{val_color}">{render_val(v)}</td>'
        html += '<td colspan="2" style="padding:5px 8px;border:1px solid var(--border2)"></td>'
        html += '</tr>'

        html += '</tbody></table></div>'

        # Pivot explanation
        if it_data["entering"] and it_data["leaving"]:
            html += (f'<div style="margin-top:6px;font-size:11px;color:#94a3b8">'
                     f'↳ Entering: <span style="color:var(--amber);font-weight:600">{it_data["entering"]}</span>'
                     f' &nbsp;|&nbsp; Leaving: <span style="color:var(--red);font-weight:600">{it_data["leaving"]}</span>'
                     f'</div>')

        html += '</div>'  # end iteration block

    # Final solution summary
    final_basis = iterations[-1]["rows"] if iterations else []
    basis_vals_html = ""
    for r in final_basis:
        basis_vals_html += f'<span style="margin-right:16px">{r["basic_var"]} = <b style="color:var(--teal)">{render_val(r["rhs"])}</b></span>'

    # actual objective value
    actual_z = iterations[-1]["z_val"] if iterations else 0
    if obj_type == "min":
        actual_z = -actual_z  # un-negate

    html += (f'<div style="background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.3);'
             f'border-radius:10px;padding:14px 18px;margin-top:8px">'
             f'<div style="font-size:11px;color:var(--indigo);font-weight:700;margin-bottom:8px">Final Basis Solution</div>'
             f'<div style="font-size:12px;color:#94a3b8;margin-bottom:6px">{basis_vals_html}</div>'
             f'<div style="font-size:13px;color:var(--amber);font-weight:700">'
             f'Z* = {render_val(round(actual_z, 4))}</div>'
             f'</div>')

    return html


# ── Pandas Sensitivity Simulation ────────────────────────────────────────────

def build_pandas_simulation(c, A, b, con_types, obj_type, n_vars, n_con, primal_res, primal_obj):
    """
    Use pandas to simulate RHS perturbations and show sensitivity table.
    """
    if not primal_res.success:
        return ("<b>Sensitivity simulation skipped</b> — primal problem infeasible.<br>"
                "Run Big-M method above for feasibility analysis.")

    records = []
    delta_vals = [-20, -10, -5, 0, 5, 10, 20]  # % perturbation of each b_i

    for i in range(n_con):
        for delta_pct in delta_vals:
            b_pert = b.copy()
            b_orig = float(b[i])
            b_new  = b_orig * (1 + delta_pct / 100) if abs(b_orig) > 1e-9 else b_orig + delta_pct * 0.1
            b_pert[i] = b_new

            # Re-standardize
            A_s = A.copy(); b_s = b_pert.copy(); ct_s = list(con_types)
            for k in range(n_con):
                if b_s[k] < 0:
                    A_s[k] *= -1; b_s[k] *= -1
                    if ct_s[k] == "le":   ct_s[k] = "ge"
                    elif ct_s[k] == "ge": ct_s[k] = "le"

            A_ub_r, b_ub_r, A_eq_r, b_eq_r = [], [], [], []
            for k in range(n_con):
                ai, bi2, ct = A_s[k], b_s[k], ct_s[k]
                if ct == "le":   A_ub_r.append(ai); b_ub_r.append(bi2)
                elif ct == "ge": A_ub_r.append(-ai); b_ub_r.append(-bi2)
                else:            A_eq_r.append(ai); b_eq_r.append(bi2)

            c_lp = -c if obj_type == "max" else c.copy()
            try:
                res = linprog(c_lp,
                              A_ub=np.array(A_ub_r) if A_ub_r else None,
                              b_ub=np.array(b_ub_r) if A_ub_r else None,
                              A_eq=np.array(A_eq_r) if A_eq_r else None,
                              b_eq=np.array(b_eq_r) if A_eq_r else None,
                              bounds=[(0, None)] * n_vars, method="highs")
                if res.success:
                    obj = float(-res.fun) if obj_type == "max" else float(res.fun)
                    feasible = True
                    x_vals = [round(float(v), 4) for v in res.x]
                else:
                    obj = None; feasible = False; x_vals = []
            except Exception:
                obj = None; feasible = False; x_vals = []

            records.append({
                "Constraint": f"b{i+1}",
                "Original b": round(b_orig, 4),
                "Δ%": f"{delta_pct:+d}%",
                "Perturbed b": round(b_new, 4),
                "Feasible": feasible,
                "Obj Z*": round(obj, 4) if obj is not None else "—",
                "ΔZ": round(obj - primal_obj, 4) if obj is not None and primal_obj is not None else "—"
            })

    df = pd.DataFrame(records)

    # Convert to HTML table
    def row_color(r):
        if not r["Feasible"]: return "rgba(239,68,68,0.08)"
        if r["Δ%"] == "+0%":  return "rgba(99,102,241,0.12)"
        return "transparent"

    html = ('<b>RHS Sensitivity Analysis (via Pandas simulation):</b><br>'
            '<p style="font-size:11px;color:var(--muted);margin:4px 0 10px">Each row perturbs one b<sub>i</sub> while keeping others fixed. '
            'ΔZ shows impact on optimal value.</p>'
            '<div style="overflow-x:auto">'
            '<table style="border-collapse:collapse;font-family:var(--mono);font-size:11px;width:100%">'
            '<thead><tr>')

    for col in df.columns:
        html += f'<th style="text-align:left;padding:6px 10px;color:var(--muted);border-bottom:1px solid var(--border);font-size:10px;text-transform:uppercase">{col}</th>'
    html += '</tr></thead><tbody>'

    prev_con = None
    for _, row in df.iterrows():
        bg = row_color(row)
        sep = "border-top:1px solid var(--border2)" if row["Constraint"] != prev_con else ""
        html += f'<tr style="background:{bg};{sep}">'
        for col in df.columns:
            val = row[col]
            extra = ""
            if col == "Feasible":
                extra = f' style="color:{"#4ade80" if val else "#f87171"};font-weight:600"'
                val = "✓ Yes" if val else "✗ No"
            elif col == "ΔZ" and isinstance(val, float):
                color = "#4ade80" if val > 0 else ("#f87171" if val < 0 else "#94a3b8")
                prefix = "+" if val > 0 else ""
                extra = f' style="color:{color};font-weight:600"'
                val = f"{prefix}{val}"
            elif col == "Δ%" and val == "+0%":
                extra = ' style="color:var(--indigo);font-weight:600"'
            html += f'<td{extra} style="padding:5px 10px;border-bottom:1px solid rgba(30,45,72,.4)">{val}</td>'
        html += '</tr>'
        prev_con = row["Constraint"]

    html += '</tbody></table></div>'
    return html


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt(v):
    v = float(v)
    if v == int(v): return str(int(v))
    return f"{v:.4g}"

def xsub(j, p="x"): return f"{p}<sub>{j+1}</sub>"

def sign_html(ct):
    return {"le": "&le;", "ge": "&ge;", "eq": "="}[ct]

def sign_plain(ct):
    return {"le": "≤", "ge": "≥", "eq": "="}[ct]


# ── Text Builders ─────────────────────────────────────────────────────────────

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
        return (f"<b>Status:</b> {res.message}<br>"
                "The primal problem has no finite optimal solution.<br>"
                '<span style="color:#f59e0b">&#9888; Big-M method applied below to diagnose infeasibility.</span>')
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
                "This may indicate numerical issues or mixed constraint types affecting classical duality.")


# ── Graph: supports 2–5 constraints ──────────────────────────────────────────

def build_graph(c, A, b, x_opt, obj_type, con_types, n_con):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.spatial import ConvexHull

        fig, ax = plt.subplots(figsize=(8.5, 7))
        fig.patch.set_facecolor("#0d0d1a")
        ax.set_facecolor("#111120")

        # Dynamic range — handle negative intercepts too
        all_intercepts_x, all_intercepts_y = [0.0], [0.0]
        for i in range(n_con):
            if abs(A[i][0]) > 1e-9: all_intercepts_x.append(abs(float(b[i]) / A[i][0]))
            if abs(A[i][1]) > 1e-9: all_intercepts_y.append(abs(float(b[i]) / A[i][1]))

        x_opt_0 = float(x_opt[0]); x_opt_1 = float(x_opt[1])
        x_max = max(max(all_intercepts_x) * 1.4, abs(x_opt_0) * 1.6, 10.0)
        y_max = max(max(all_intercepts_y) * 1.4, abs(x_opt_1) * 1.6, 10.0)
        x_max = min(x_max, 500); y_max = min(y_max, 500)

        # Allow slight negative display
        x_min_plot = min(0.0, x_opt_0 * 1.2) - 0.5
        y_min_plot = min(0.0, x_opt_1 * 1.2) - 0.5

        x_vals = np.linspace(x_min_plot - 2, x_max + 2, 1200)

        palette = ["#7c6af7", "#f7a26a", "#4ade80", "#f87171", "#38bdf8",
                   "#e879f9", "#fbbf24", "#a78bfa"]

        # Plot constraint lines
        for i in range(n_con):
            a0, a1, bi = float(A[i][0]), float(A[i][1]), float(b[i])
            col = palette[i % len(palette)]
            lbl = f"C{i+1}: {fmt(a0)}x₁+{fmt(a1)}x₂ {sign_plain(con_types[i])} {fmt(bi)}"
            if abs(a1) > 1e-9:
                y_line = (bi - a0 * x_vals) / a1
                mask   = (y_line >= y_min_plot - 2) & (y_line <= y_max * 1.6)
                y_plot = np.where(mask, y_line, np.nan)
                ax.plot(x_vals, y_plot, color=col, lw=2, label=lbl, zorder=4)
            elif abs(a0) > 1e-9:
                xv = bi / a0
                ax.axvline(xv, color=col, lw=2, label=lbl, zorder=4)

        # Feasible region
        verts = feasible_vertices(A, b, x_max, y_max, con_types)
        if len(verts) >= 3:
            pts = np.array(verts)
            try:
                hull = ConvexHull(pts)
                poly = pts[hull.vertices]
                ax.fill(poly[:, 0], poly[:, 1], color="#7c6af7", alpha=0.13,
                         zorder=1, label="Feasible region")
                for pt in poly:
                    ax.plot(pt[0], pt[1], "o", color="#7c6af7",
                             markersize=5, alpha=0.6, zorder=5)
            except Exception:
                pass

        # Objective iso-line
        if abs(c[1]) > 1e-9:
            obj_val = c[0] * x_opt_0 + c[1] * x_opt_1
            y_obj   = (obj_val - c[0] * x_vals) / c[1]
            mask    = (y_obj >= y_min_plot - 2) & (y_obj <= y_max * 1.6)
            ax.plot(x_vals, np.where(mask, y_obj, np.nan), "--",
                     color="#fbbf24", lw=1.5, alpha=0.7,
                     label=f"Z={fmt(obj_val)} (objective)", zorder=3)

        # Gradient arrow
        norm = math.sqrt(float(c[0]) ** 2 + float(c[1]) ** 2)
        if norm > 1e-9:
            dx, dy = float(c[0]) / norm, float(c[1]) / norm
            scale  = min(x_max, y_max) * 0.14
            sd     = 1 if obj_type == "max" else -1
            ax.annotate("", xy=(x_opt_0 + sd * dx * scale, x_opt_1 + sd * dy * scale),
                         xytext=(x_opt_0, x_opt_1),
                         arrowprops=dict(arrowstyle="-|>", color="#fbbf24", lw=2), zorder=8)

        # Optimal point
        ax.scatter([x_opt_0], [x_opt_1], color="#fbbf24", s=220, zorder=10,
                    edgecolors="white", linewidths=1.5,
                    label=f"Optimal: ({fmt(x_opt_0)}, {fmt(x_opt_1)})")
        ax.annotate(f"  Z*={fmt(c[0]*x_opt_0+c[1]*x_opt_1)}",
                     (x_opt_0, x_opt_1), color="#fbbf24", fontsize=9, zorder=11,
                     xytext=(6, 6), textcoords="offset points")

        ax.set_xlim(x_min_plot, x_max)
        ax.set_ylim(y_min_plot, y_max)
        ax.set_xlabel("x₁", color="#ccc", fontsize=12)
        ax.set_ylabel("x₂", color="#ccc", fontsize=12)
        ax.tick_params(colors="#666")
        for sp in ax.spines.values(): sp.set_edgecolor("#222244")
        ax.grid(True, color="#1e1e38", lw=0.6, linestyle="--")
        ax.axhline(0, color="#334466", lw=0.8, zorder=0)
        ax.axvline(0, color="#334466", lw=0.8, zorder=0)

        # Legend — place outside if many constraints
        ncol = 2 if n_con >= 4 else 1
        ax.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e",
                   edgecolor="#333", labelcolor="white", framealpha=0.92,
                   ncol=ncol, bbox_to_anchor=(1, 1))

        ax.set_title(
            f"{'Maximisation' if obj_type=='max' else 'Minimisation'} — "
            f"Graphical Method ({n_con} constraints)",
            color="#e8e8f0", fontsize=12, pad=10)
        fig.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                     facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print("Graph error:", e)
        return None


def build_big_m_graph(c, A, b, x_opt, obj_type, con_types, n_con):
    """Separate graph labelled as Big-M recovery solution."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.spatial import ConvexHull

        fig, ax = plt.subplots(figsize=(8.5, 7))
        fig.patch.set_facecolor("#0d0d1a")
        ax.set_facecolor("#111120")

        x_opt_0 = float(x_opt[0]); x_opt_1 = float(x_opt[1])
        x_max = max(abs(x_opt_0) * 2, 10.0)
        y_max = max(abs(x_opt_1) * 2, 10.0)
        for i in range(n_con):
            if abs(A[i][0]) > 1e-9: x_max = max(x_max, abs(float(b[i]) / A[i][0]) * 1.4)
            if abs(A[i][1]) > 1e-9: y_max = max(y_max, abs(float(b[i]) / A[i][1]) * 1.4)
        x_max = min(x_max, 500); y_max = min(y_max, 500)
        x_min_plot = min(0.0, x_opt_0 * 1.2) - 0.5
        y_min_plot = min(0.0, x_opt_1 * 1.2) - 0.5

        x_vals  = np.linspace(x_min_plot - 2, x_max + 2, 1200)
        palette = ["#7c6af7","#f7a26a","#4ade80","#f87171","#38bdf8","#e879f9","#fbbf24"]

        for i in range(n_con):
            a0, a1, bi = float(A[i][0]), float(A[i][1]), float(b[i])
            col = palette[i % len(palette)]
            lbl = f"C{i+1}: {fmt(a0)}x₁+{fmt(a1)}x₂ {sign_plain(con_types[i])} {fmt(bi)}"
            if abs(a1) > 1e-9:
                y_line = (bi - a0 * x_vals) / a1
                mask   = (y_line >= y_min_plot - 2) & (y_line <= y_max * 1.6)
                ax.plot(x_vals, np.where(mask, y_line, np.nan), color=col, lw=2, label=lbl, zorder=4)
            elif abs(a0) > 1e-9:
                ax.axvline(bi / a0, color=col, lw=2, label=lbl, zorder=4)

        ax.scatter([x_opt_0], [x_opt_1], color="#f87171", s=240, zorder=10,
                    edgecolors="white", linewidths=2,
                    label=f"Big-M Soln: ({fmt(x_opt_0)}, {fmt(x_opt_1)})", marker="*")
        ax.annotate(f"  Z*≈{fmt(c[0]*x_opt_0+c[1]*x_opt_1)}",
                     (x_opt_0, x_opt_1), color="#f87171", fontsize=9, zorder=11,
                     xytext=(6, 6), textcoords="offset points")

        ax.set_xlim(x_min_plot, x_max); ax.set_ylim(y_min_plot, y_max)
        ax.set_xlabel("x₁", color="#ccc", fontsize=12); ax.set_ylabel("x₂", color="#ccc", fontsize=12)
        ax.tick_params(colors="#666")
        for sp in ax.spines.values(): sp.set_edgecolor("#222244")
        ax.grid(True, color="#1e1e38", lw=0.6, linestyle="--")
        ax.axhline(0, color="#334466", lw=0.8, zorder=0); ax.axvline(0, color="#334466", lw=0.8, zorder=0)
        ncol = 2 if n_con >= 4 else 1
        ax.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e", edgecolor="#333",
                   labelcolor="white", framealpha=0.92, ncol=ncol)
        ax.set_title("Big-M Recovery Solution — Constraint Lines", color="#e8e8f0", fontsize=12, pad=10)
        fig.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print("Big-M graph error:", e)
        return None


def feasible_vertices(A, b, x_max, y_max, con_types):
    """Find all vertices of feasible region for 2-variable LP."""
    extra_A  = [[-1, 0], [0, -1], [1, 0], [0, 1]]
    extra_b  = [0, 0, x_max, y_max]
    extra_ct = ["le", "le", "le", "le"]
    rows = list(A) + extra_A
    rhs  = list(b) + extra_b
    cts  = list(con_types) + extra_ct
    pts  = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            try:
                M_mat = np.array([rows[i], rows[j]], dtype=float)
                if abs(np.linalg.det(M_mat)) < 1e-10: continue
                pt = np.linalg.solve(M_mat, [rhs[i], rhs[j]])
                if pt[0] < -1e-6 or pt[1] < -1e-6: continue
                if pt[0] > x_max * 1.01 or pt[1] > y_max * 1.01: continue
                feasible = True
                for k in range(len(A)):
                    lhs_k = A[k][0] * pt[0] + A[k][1] * pt[1]
                    ct = con_types[k]
                    if ct == "le" and lhs_k > b[k] + 1e-6: feasible = False; break
                    if ct == "ge" and lhs_k < b[k] - 1e-6: feasible = False; break
                    if ct == "eq" and abs(lhs_k - b[k]) > 1e-6: feasible = False; break
                if feasible:
                    pts.append(pt.tolist())
            except Exception:
                pass
    return pts



if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)