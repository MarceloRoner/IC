from flask import Flask, render_template, request
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg')  # permite uso do matplotlib sem ambiente gráfico
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)

def newton_method_symbolic(equation_str, x0, tol=1e-6, max_iter=100, verbose=False):
    x = sp.symbols('x')
    try:
        f_sym = sp.sympify(equation_str)
    except Exception as e:
        return None, False, 0, f"Erro na interpretação da equação: {e}", []
    df_sym = sp.diff(f_sym, x)
    f = sp.lambdify(x, f_sym, 'numpy')
    df = sp.lambdify(x, df_sym, 'numpy')

    log = []
    approximations = []  # guardar cada x_i
    x_current = x0

    for i in range(1, max_iter + 1):
        try:
            f_val = f(x_current)
            df_val = df(x_current)
        except Exception as e:
            return x_current, False, i, f"Erro ao avaliar a função: {e}", approximations
        
        approximations.append(x_current)

        if abs(df_val) < 1e-12:
            log.append(f"Iteração {i}: Derivada muito próxima de zero em x = {x_current:.6f}.")
            return x_current, False, i, "\n".join(log), approximations

        x_next = x_current - f_val / df_val

        if verbose:
            log.append(
                f"Iteração {i}: x = {x_current:.6f}, "
                f"f(x) = {f_val:.6f}, f'(x) = {df_val:.6f}, "
                f"Próximo x = {x_next:.6f}"
            )

        if abs(x_next - x_current) < tol:
            approximations.append(x_next)
            return x_next, True, i, "\n".join(log), approximations

        x_current = x_next

    approximations.append(x_current)
    return x_current, False, max_iter, "\n".join(log), approximations


@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    log_text = ''
    plot_url = ''

    if request.method == 'POST':
        equation = request.form.get('equation')
        x0 = float(request.form.get('x0'))
        tol = float(request.form.get('tol'))
        max_iter = int(request.form.get('max_iter'))
        verbose = 'verbose' in request.form

        root, converged, iters, log_text, approximations = newton_method_symbolic(
            equation, x0, tol, max_iter, verbose
        )

        if converged:
            result = f"Convergência alcançada: raiz = {root:.6f} em {iters} iterações."
        else:
            result = f"Não convergiu em {iters} iterações. Última aproximação: {root:.6f}"

        # Gera o gráfico de erro (|x_{n+1} - x_n|) em escala log
        if len(approximations) > 1:
            errors = []
            for i in range(len(approximations) - 1):
                errors.append(abs(approximations[i+1] - approximations[i]))

            fig, ax = plt.subplots(figsize=(6,4))
            ax.set_title("Evolução do Erro a cada Iteração (Método de Newton)")
            ax.set_xlabel("Iteração")
            ax.set_ylabel("Erro (|x_{n+1} - x_n|)")
            ax.set_yscale("log")
            ax.grid(True, which='both', ls='--')

            ax.plot(range(1, len(errors) + 1), errors, marker='o', linestyle='-')
            
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            # Codifica em base64 para embutir no HTML
            plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)

    return render_template(
        'index.html',
        result=result,
        log=log_text,
        plot_url=plot_url
    )


if __name__ == '__main__':
    app.run(debug=True)
