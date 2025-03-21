import tkinter as tk
from tkinter import ttk
import sympy as sp
import numpy as np

def newton_method_symbolic(equation_str, x0, tol=1e-6, max_iter=100, verbose=False, output_callback=None):
    x = sp.symbols('x')
    try:
        f_sym = sp.sympify(equation_str)
    except Exception as e:
        if output_callback:
            output_callback(f"Erro na interpretação da equação: {e}")
        return None, False, 0
    df_sym = sp.diff(f_sym, x)
    f = sp.lambdify(x, f_sym, 'numpy')
    df = sp.lambdify(x, df_sym, 'numpy')
    if verbose and output_callback:
        output_callback(f"Equação: {f_sym}")
        output_callback(f"Derivada: {df_sym}")
        output_callback("-" * 40)
    x_current = x0
    for i in range(1, max_iter + 1):
        try:
            f_val = f(x_current)
            df_val = df(x_current)
        except Exception as e:
            if output_callback:
                output_callback(f"Erro ao avaliar a função: {e}")
            return x_current, False, i
        if abs(df_val) < 1e-12:
            if verbose and output_callback:
                output_callback(f"Iteração {i}: Derivada muito próxima de zero em x = {x_current:.6f}.")
            return x_current, False, i
        x_next = x_current - f_val / df_val
        if verbose and output_callback:
            output_callback(f"Iteração {i}:")
            output_callback(f"  x = {x_current:.6f}, f(x) = {f_val:.6f}, f'(x) = {df_val:.6f}")
            output_callback(f"  Próximo x = {x_next:.6f}")
            output_callback("-" * 40)
        if abs(x_next - x_current) < tol:
            return x_next, True, i
        x_current = x_next
    return x_current, False, max_iter

def run_newton():
    eq_str = equation_entry.get()
    try:
        x0 = float(x0_entry.get())
    except ValueError:
        output_text.insert(tk.END, "Erro: chute inicial inválido.\n")
        return
    try:
        tol = float(tol_entry.get())
    except ValueError:
        output_text.insert(tk.END, "Erro: tolerância inválida.\n")
        return
    try:
        max_iter = int(max_iter_entry.get())
    except ValueError:
        output_text.insert(tk.END, "Erro: número máximo de iterações inválido.\n")
        return
    verbose = verbose_var.get()
    output_text.delete(1.0, tk.END)
    def gui_output(msg):
        output_text.insert(tk.END, msg + "\n")
        output_text.see(tk.END)
    result, converged, iters = newton_method_symbolic(eq_str, x0, tol, max_iter, verbose, gui_output)
    if result is None:
        return
    if converged:
        gui_output(f"\nConvergência alcançada: raiz = {result:.6f} em {iters} iterações.")
    else:
        gui_output(f"\nNão convergiu em {iters} iterações. Última aproximação: {result:.6f}")

root = tk.Tk()
root.title("Método de Newton-Raphson")
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) # type: ignore
equation_label = ttk.Label(frame, text="Equação f(x):")
equation_label.grid(row=0, column=0, sticky=tk.W)
equation_entry = ttk.Entry(frame, width=30)
equation_entry.grid(row=0, column=1, sticky=(tk.W, tk.E)) # type: ignore
equation_entry.insert(0, "x**2 - 2")
x0_label = ttk.Label(frame, text="Chute Inicial x0:")
x0_label.grid(row=1, column=0, sticky=tk.W)
x0_entry = ttk.Entry(frame, width=30)
x0_entry.grid(row=1, column=1, sticky=(tk.W, tk.E)) # type: ignore
x0_entry.insert(0, "1.0")
tol_label = ttk.Label(frame, text="Tolerância:")
tol_label.grid(row=2, column=0, sticky=tk.W)
tol_entry = ttk.Entry(frame, width=30)
tol_entry.grid(row=2, column=1, sticky=(tk.W, tk.E)) # type: ignore
tol_entry.insert(0, "1e-6")
max_iter_label = ttk.Label(frame, text="Máximo de Iterações:")
max_iter_label.grid(row=3, column=0, sticky=tk.W)
max_iter_entry = ttk.Entry(frame, width=30)
max_iter_entry.grid(row=3, column=1, sticky=(tk.W, tk.E)) # type: ignore
max_iter_entry.insert(0, "50")
verbose_var = tk.BooleanVar()
verbose_check = ttk.Checkbutton(frame, text="Exibir detalhes das iterações", variable=verbose_var)
verbose_check.grid(row=4, column=0, columnspan=2, sticky=tk.W)
run_button = ttk.Button(frame, text="Calcular", command=run_newton)
run_button.grid(row=5, column=0, columnspan=2, pady=10)
output_label = ttk.Label(frame, text="Resultado:")
output_label.grid(row=6, column=0, sticky=tk.W)
output_text = tk.Text(frame, width=50, height=15)
output_text.grid(row=7, column=0, columnspan=2, pady=5)
root.mainloop()
