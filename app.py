import os
import io
import base64
from functools import lru_cache
from typing import Tuple, List, Optional

from flask import Flask, render_template, request, send_file
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
import numpy as np
import matplotlib
from weasyprint import HTML

# Permite usar o Matplotlib em ambientes sem servidor X (Render, Heroku, etc.)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

###############################################################################
# Configuração mínima (continua tudo em um único arquivo)
###############################################################################

class Config:
    """Configuração básica.  Use variáveis de ambiente para produção."""

    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev")


app = Flask(__name__)
app.config.from_object(Config)

###############################################################################
# Utilidades
###############################################################################

X = sp.symbols("x")
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


@lru_cache(maxsize=32)
def _build_functions(equation: str) -> Tuple[sp.Expr, callable, callable]:
    """Compila f(x) e f'(x) a partir da string *equation* (com cache)."""
    try:
        expr = parse_expr(
            equation,
            transformations=TRANSFORMATIONS,
            local_dict={"x": X},
            evaluate=True,
        )
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Erro ao interpretar a equação: {exc}") from exc

    f = sp.lambdify(X, expr, "numpy")
    df = sp.lambdify(X, sp.diff(expr, X), "numpy")
    return expr, f, df


def fig_to_base64(fig: "plt.Figure") -> str:
    """Converte uma figura Matplotlib para string base64."""
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return encoded

###############################################################################
# Núcleo numérico: Método de Newton
###############################################################################

def newton_method(
    equation: str,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    verbose: bool = False,
) -> Tuple[Optional[float], bool, int, str, List[float]]:
    """Resolve *f(x)=0* pelo Método de Newton‑Raphson.

    Retorna (raiz|None, convergiu?, nº iterações, log, lista de aproximações).
    """
    try:
        _expr, f, df = _build_functions(equation)
    except ValueError as err:
        return None, False, 0, str(err), []

    log_lines: List[str] = []
    approximations: List[float] = []
    x_curr: float = x0

    for i in range(1, max_iter + 1):
        try:
            f_val = float(f(x_curr))
            df_val = float(df(x_curr))
        except (TypeError, ValueError, OverflowError) as err:
            return x_curr, False, i, f"Erro numérico: {err}", approximations

        approximations.append(x_curr)

        if abs(df_val) < 1e-12:
            log_lines.append(
                f"Iteração {i}: derivada ≈ 0 em x = {x_curr:.6g}. Método parou."
            )
            return x_curr, False, i, "\n".join(log_lines), approximations

        x_next = x_curr - f_val / df_val

        if verbose:
            log_lines.append(
                f"Iteração {i}: x = {x_curr:.6g}, f(x) = {f_val:.6g}, f'(x) = {df_val:.6g} → xₙ₊₁ = {x_next:.6g}"
            )

        if abs(x_next - x_curr) < tol:
            approximations.append(x_next)
            return x_next, True, i, "\n".join(log_lines), approximations

        x_curr = x_next

    return x_curr, False, max_iter, "\n".join(log_lines), approximations

###############################################################################
# Rotas Flask
###############################################################################

@app.route("/", methods=["GET", "POST"])
def index():
    # ---------------- estado inicial (GET) ----------------
    result: str = ""
    log_text: str = ""
    plots: List[str] = []
    equation: str = ""          # ← garante existência no primeiro acesso

    # ---------------- POST: cálculo -----------------------
    if request.method == "POST":
        equation = request.form.get("equation", "").strip()
        try:
            x0 = float(request.form.get("x0", "0"))
            tol = float(request.form.get("tol", "1e-6"))
            max_iter = int(request.form.get("max_iter", "100"))
        except ValueError:
            result = "Erro: verifique se os valores numéricos estão corretos."
            return render_template(
                "index.html", result=result, log=log_text, plots=plots, equation=equation
            )

        verbose   = "verbose"   in request.form
        plot_fx   = "plot_fx"   in request.form
        plot_iter = "plot_iter" in request.form
        plot_fxn  = "plot_fxn"  in request.form
        plot_error= "plot_error"in request.form

        root, converged, iters, log_text, approximations = newton_method(
            equation, x0, tol, max_iter, verbose
        )

        result = (
            f"Convergência alcançada: raiz ≈ {root:.6g} em {iters} iterações."
            if converged and root is not None
            else (
                f"Não convergiu em {iters} iterações. Última aproximação: {root:.6g}"
                if root is not None else "Não foi possível interpretar a equação."
            )
        )

        try:
            _expr, f, _df = _build_functions(equation)
        except ValueError:
            f = lambda _x: np.nan  # type: ignore

        # ----- gráficos -----
        if plot_fx:
            x_vals = np.linspace(x0 - 5, x0 + 5, 400)
            y_vals = f(x_vals)
            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals); ax.axhline(0, color="gray", ls="--")
            ax.set_title("f(x) na vizinhança de x₀"); ax.grid(True)
            plots.append(fig_to_base64(fig))

        if plot_iter and approximations:
            fig, ax = plt.subplots()
            ax.plot(range(len(approximations)), approximations, marker="o")
            ax.set_xlabel("Iteração"); ax.set_ylabel("xₙ")
            ax.set_title("Aproximações sucessivas"); ax.grid(True)
            plots.append(fig_to_base64(fig))

        if plot_fxn and approximations:
            y_fxn = [f(xi) for xi in approximations]
            fig, ax = plt.subplots()
            ax.plot(range(len(y_fxn)), y_fxn, marker="x")
            ax.set_xlabel("Iteração"); ax.set_ylabel("f(xₙ)")
            ax.set_title("Valores de f(xₙ)"); ax.grid(True)
            plots.append(fig_to_base64(fig))

        if plot_error and len(approximations) > 1:
            errors = [abs(approximations[i+1]-approximations[i]) for i in range(len(approximations)-1)]
            fig, ax = plt.subplots()
            ax.plot(range(1,len(errors)+1), errors, marker="o")
            ax.set_yscale("log"); ax.set_xlabel("Iteração"); ax.set_ylabel("Erro absoluto")
            ax.set_title("Evolução do erro (log)"); ax.grid(True, which="both", ls="--")
            plots.append(fig_to_base64(fig))

    # ---------------- resposta (GET ou POST) ---------------
    return render_template(
        "index.html",
        result=result,
        log=log_text,
        plots=plots,
        equation=equation,   # ← agora o template recebe a equação
    )

###############################################################################
# Exportação de relatório PDF
###############################################################################

@app.route("/pdf", methods=["POST"])
def gerar_pdf():
    """Recebe dados do formulário oculto e devolve um PDF."""

    equation = request.form.get("equation", "")
    result = request.form.get("result", "")
    log = request.form.get("log", "")
    plots = request.form.getlist("plots")

    html_str = render_template(
        "relatorio.html",
        equation=equation,
        result=result,
        log=log,
        plots=plots,
    )
    pdf_bytes = HTML(string=html_str, base_url=request.host_url).write_pdf()
    return send_file(
        io.BytesIO(pdf_bytes),
        download_name="relatorio.pdf",
        mimetype="application/pdf",
    )

###############################################################################
# Execução
###############################################################################

if __name__ == "__main__":
    debug_mode = bool(int(os.getenv("FLASK_DEBUG", "1")))
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug_mode)
