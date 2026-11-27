"""
Módulo de Verificación Simbólica: verificacion_sympy.py
Comprobación analítica exacta mediante SymPy de derivadas parciales,
vector gradiente, derivada direccional, primeras 2 iteraciones de
Descenso del Gradiente y error relativo frente a diferencias finitas.
"""

import numpy as np
import sympy as sp
from calculo_vectorial import gradiente
from main import P0, META, K_A, OBSTACULOS, V_PRUEBA, ALPHA, potencial_U


def construir_modelo_simbolico():
    """Construye U(x, y) y sus derivadas parciales exactas con SymPy."""
    x, y = sp.symbols("x y", real=True)

    atraccion = (
        sp.Rational(1, 2)
        * sp.nsimplify(K_A)
        * ((x - sp.nsimplify(META[0])) ** 2 + (y - sp.nsimplify(META[1])) ** 2)
    )

    repulsion = sp.Integer(0)
    for obs in OBSTACULOS:
        xo = sp.nsimplify(obs["centro"][0])
        yo = sp.nsimplify(obs["centro"][1])
        amp = sp.nsimplify(obs["A"])
        sigma = sp.nsimplify(obs["sigma"])
        repulsion += amp * sp.exp(-((x - xo) ** 2 + (y - yo) ** 2) / (2 * sigma**2))

    u_sym = atraccion + repulsion
    du_dx_sym = sp.diff(u_sym, x)
    du_dy_sym = sp.diff(u_sym, y)

    return x, y, u_sym, du_dx_sym, du_dy_sym


def ejecutar_verificacion_analitica() -> None:
    """Evalúa los valores exactos y compara contra el método numérico."""
    x, y, u_sym, du_dx_sym, du_dy_sym = construir_modelo_simbolico()

    print("=" * 76)
    print("VERIFICACIÓN ANALÍTICA EXACTA CON SYMPY (INTEGRANTE 2)")
    print("=" * 76)

    print("\n--- 1. EXPRESIONES SIMBÓLICAS EXACTAS ---")
    print(f"U(x, y)  = {u_sym}")
    print(f"dU/dx    = {du_dx_sym}")
    print(f"dU/dy    = {du_dy_sym}")

    # Evaluación exacta en P0
    subs_p0 = {x: sp.nsimplify(P0[0]), y: sp.nsimplify(P0[1])}
    u_p0_exacto = float(u_sym.subs(subs_p0).evalf(25))
    gx_p0_exacto = float(du_dx_sym.subs(subs_p0).evalf(25))
    gy_p0_exacto = float(du_dy_sym.subs(subs_p0).evalf(25))
    grad_p0_exacto = np.array([gx_p0_exacto, gy_p0_exacto], dtype=float)
    norma_grad_p0 = float(np.linalg.norm(grad_p0_exacto))

    # Derivada direccional exacta en P0
    v_arr = np.array(V_PRUEBA, dtype=float)
    u_unit = v_arr / np.linalg.norm(v_arr)
    du_p0_exacto = float(np.dot(grad_p0_exacto, u_unit))

    print("\n--- 2. VALORES EXACTOS EN P0 ---")
    print(f"U(P0) exacta           = {u_p0_exacto:.10f}")
    print(f"nabla_U(P0) exacto     = [{gx_p0_exacto:.10f}, {gy_p0_exacto:.10f}]")
    print(f"||nabla_U(P0)|| exacto = {norma_grad_p0:.10f}")
    print(f"Vector unitario u      = [{u_unit[0]:.10f}, {u_unit[1]:.10f}]")
    print(f"D_u U(P0) exacta       = {du_p0_exacto:.10f}")

    # Primeras 2 iteraciones analíticas exactas (P1 y P2)
    p0_arr = np.array(P0, dtype=float)
    p1_exacto = p0_arr - ALPHA * grad_p0_exacto
    u_p1_exacto = float(
        u_sym.subs({x: sp.Float(p1_exacto[0], 25), y: sp.Float(p1_exacto[1], 25)}).evalf(25)
    )

    subs_p1 = {x: sp.Float(p1_exacto[0], 25), y: sp.Float(p1_exacto[1], 25)}
    grad_p1_exacto = np.array(
        [
            float(du_dx_sym.subs(subs_p1).evalf(25)),
            float(du_dy_sym.subs(subs_p1).evalf(25)),
        ],
        dtype=float,
    )

    p2_exacto = p1_exacto - ALPHA * grad_p1_exacto
    u_p2_exacto = float(
        u_sym.subs({x: sp.Float(p2_exacto[0], 25), y: sp.Float(p2_exacto[1], 25)}).evalf(25)
    )

    print(f"\n--- 3. PRIMERAS 2 ITERACIONES ANALÍTICAS (alpha = {ALPHA}) ---")
    print(f"Iteración 1 -> P1 = ({p1_exacto[0]:.10f}, {p1_exacto[1]:.10f}) | U(P1) = {u_p1_exacto:.10f}")
    print(f"nabla_U(P1)       = [{grad_p1_exacto[0]:.10f}, {grad_p1_exacto[1]:.10f}]")
    print(f"Iteración 2 -> P2 = ({p2_exacto[0]:.10f}, {p2_exacto[1]:.10f}) | U(P2) = {u_p2_exacto:.10f}")

    # Tabla comparativa de error relativo al variar h
    print("\n--- 4. TABLA DE ERROR RELATIVO EN nabla_U(P0) AL VARIAR h ---")
    print(f"{'Paso h':<10} | {'dU/dx Numérico':<16} | {'dU/dy Numérico':<16} | {'Error Relativo (%)':<18}")
    print("-" * 76)

    for h_val in [1e-1, 1e-3, 1e-5, 1e-7]:
        grad_num = gradiente(potencial_U, P0[0], P0[1], h=h_val)
        err_rel = (
            np.linalg.norm(grad_num - grad_p0_exacto) / np.linalg.norm(grad_p0_exacto)
        ) * 100.0
        print(
            f"{h_val:<10.0e} | {grad_num[0]:<16.10f} | {grad_num[1]:<16.10f} | {err_rel:<18.10e}"
        )
    print("=" * 76)


if __name__ == "__main__":
    ejecutar_verificacion_analitica()