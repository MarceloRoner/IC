import sympy as sp
import numpy as np

"""
convergência quadrática
"""

def newton_method_symbolic(equation_str, x0, tol=1e-6, max_iter=100, verbose=False):
    """
    Encontra a raiz de uma equação usando o método de Newton-Raphson.
    A equação é fornecida como string e a derivada é calculada simbolicamente.

    Parâmetros:
        equation_str (str): A equação em função de x, por exemplo, "x**2 - 2".
        x0 (float): Chute inicial.
        tol (float, optional): Tolerância para a convergência. Padrão: 1e-6.
        max_iter (int, optional): Número máximo de iterações. Padrão: 100.
        verbose (bool, optional): Se True, exibe os detalhes de cada iteração.

    Retorna:
        tuple: (raiz, convergiu, iterações)
            raiz (float): Aproximação da raiz.
            convergiu (bool): True se convergiu, False caso contrário.
            iterações (int): Número de iterações realizadas.
    """
    # Define a variável simbólica
    x = sp.symbols('x')
    
    # Converte a string da equação em uma expressão simbólica
    f_sym = sp.sympify(equation_str)
    
    # Calcula a derivada simbolicamente
    df_sym = sp.diff(f_sym, x)
    
    # Converte as expressões simbólicas em funções numéricas (para numpy)
    f = sp.lambdify(x, f_sym, 'numpy')
    df = sp.lambdify(x, df_sym, 'numpy')
    
    if verbose:
        print("Equação fornecida: f(x) =", f_sym)
        print("Derivada calculada: f'(x) =", df_sym)
        print("-" * 50)
    
    # Processo iterativo do método de Newton-Raphson
    x_current = x0
    for i in range(1, max_iter + 1):
        f_val = f(x_current)
        df_val = df(x_current)
        
        # Evita divisão por zero ou derivada muito pequena
        if abs(df_val) < 1e-12:
            if verbose:
                print(f"Iteração {i}: Derivada muito próxima de zero em x = {x_current:.6f}.")
            return x_current, False, i
        
        # Calcula o próximo valor
        x_next = x_current - f_val / df_val
        
        if verbose:
            print(f"Iteração {i}:")
            print(f"  x = {x_current:.6f}, f(x) = {f_val:.6f}, f'(x) = {df_val:.6f}")
            print(f"  Próximo x = {x_next:.6f}")
            print("-" * 50)
        
        # Verifica convergência com base na mudança entre iterações
        if abs(x_next - x_current) < tol:
            return x_next, True, i
        
        x_current = x_next

    return x_current, False, max_iter

if __name__ == '__main__':
    # Solicita a equação e os parâmetros ao usuário
    eq_str = input("Digite a equação em função de x (ex: x**2 - 2): ")
    x0 = float(input("Digite o chute inicial: "))
    tol = float(input("Digite a tolerância (ex: 1e-6): "))
    max_iter = int(input("Digite o número máximo de iterações: "))
    verbose_input = input("Exibir detalhes das iterações? (s/n): ")
    verbose = verbose_input.strip().lower() == 's'
    
    # Executa o método
    raiz, convergiu, iteracoes = newton_method_symbolic(eq_str, x0, tol, max_iter, verbose)
    
    # Exibe o resultado final
    if convergiu:
        print(f"\nConvergência alcançada: raiz = {raiz:.6f} em {iteracoes} iterações.")
    else:
        print(f"\nNão convergiu em {iteracoes} iterações. Última aproximação: {raiz:.6f}")
