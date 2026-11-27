"""
Script Principal: main.py
Caso de Estudio — Opción B: Navegación Autónoma mediante Campos Potenciales (Robótica)
Proyecto Final de Cálculo Vectorial — UP Chiapas
"""

import numpy as np
from calculo_vectorial import (
    gradiente,
    derivada_direccional,
    verificar_teorema_direccional,
    evaluar_sensores_agente,
    descenso_gradiente,
)

# ==============================================================================
# PARÁMETROS CONFIGURABLES (MODIFICAR AQUÍ DURANTE LA DEFENSA EN VIVO)
# ==============================================================================
P0 = (0.0, 0.0)                      # Punto inicial del agente autónomo
META = (5.0, 5.0)                    # Mínimo global (destino)
K_A = 1.0                            # Coeficiente de atracción parabólica

OBSTACULOS = [
    {"centro": (2.0, 2.6), "A": 4.0, "sigma": 0.8},  # Obstáculo 1 (máximo local)
    {"centro": (3.4, 1.8), "A": 3.5, "sigma": 0.7},  # Obstáculo 2 (máximo local)
]

V_PRUEBA = (3.0, 4.0)                # Vector dirección de prueba (||v|| = 5.0 -> u = [0.6, 0.8])
ALPHA = 0.08                         # Tasa de aprendizaje principal
ALPHAS_COMPARACION = [0.05, 0.08, 0.45]  # Tasas para la gráfica de convergencia
H_DEFAULT = 1e-5                     # Tamaño de paso para diferencias finitas centradas
TOL = 1e-4                           # Tolerancia de paro para ||nabla U||
MAX_ITER = 1000                      # Límite máximo de iteraciones
# ==============================================================================


def potencial_U(x: float, y: float) -> float:
    """
    Evalúa la superficie de potencial artificial U(x, y):
        U(x, y) = 0.5 * k_a * ((x - x_g)^2 + (y - y_g)^2)
                  + sum(A_i * exp(-((x - x_oi)^2 + (y - y_oi)^2) / (2 * sigma_i^2)))
    """
    atraccion = 0.5 * K_A * ((x - META[0]) ** 2 + (y - META[1]) ** 2)
    repulsion = 0.0
    for obs in OBSTACULOS:
        xo, yo = obs["centro"]
        amp = obs["A"]
        sigma = obs["sigma"]
        distancia_cuad = (x - xo) ** 2 + (y - yo) ** 2
        repulsion += amp * np.exp(-distancia_cuad / (2.0 * sigma ** 2))
    return float(atraccion + repulsion)


def imprimir_resumen_numerico() -> None:
    """Ejecuta y muestra en consola las validaciones matemáticas del sistema."""
    print("=" * 72)
    print("SIMULADOR DE NAVEGACIÓN AUTÓNOMA MEDIANTE CAMPOS POTENCIALES U(x, y)")
    print("=" * 72)
    print(f"Punto inicial P0       : {P0}")
    print(f"Meta (mínimo global)   : {META} (k_a = {K_A})")
    print(f"Vector prueba v        : {V_PRUEBA}")
    print(f"Tasa de aprendizaje    : alpha = {ALPHA}, paso h = {H_DEFAULT}\n")

    # 1. Gradiente y Derivada Direccional en P0
    grad_p0 = gradiente(potencial_U, P0[0], P0[1], H_DEFAULT)
    norma_grad = np.linalg.norm(grad_p0)
    du_p0, u_norm = derivada_direccional(potencial_U, P0, V_PRUEBA, H_DEFAULT)

    print("--- 1. EVALUACIÓN EN EL PUNTO INICIAL P0 ---")
    print(f"U(P0)                  = {potencial_U(*P0):.6f}")
    print(f"Gradiente nabla_U(P0)  = [{grad_p0[0]:.6f}, {grad_p0[1]:.6f}]")
    print(f"||nabla_U(P0)||        = {norma_grad:.6f}")
    print(f"Vector unitario u      = [{u_norm[0]:.6f}, {u_norm[1]:.6f}]")
    print(f"Derivada direccional   = D_u U(P0) = {du_p0:.6f}\n")

    # 2. Comprobación computacional del teorema fundamental
    res_teorema = verificar_teorema_direccional(potencial_U, P0, n_muestras=200, h=H_DEFAULT)
    print("--- 2. VERIFICACIÓN DEL TEOREMA (-||nabla U|| <= D_u U <= ||nabla U||) ---")
    print(f"Cotas teóricas         : [{res_teorema['cota_inferior']:.6f}, {res_teorema['cota_superior']:.6f}]")
    print(f"Rango en 200 muestras  : [{res_teorema['min_observado']:.6f}, {res_teorema['max_observado']:.6f}]")
    print(f"¿Cumple el teorema?    : {res_teorema['cumple_teorema']}\n")

    # 3. Evaluación de sensores virtuales del agente autónomo en P0
    mejor_u, menor_du, u_teorico = evaluar_sensores_agente(potencial_U, P0, n_direcciones=36, h=H_DEFAULT)
    print("--- 3. SENSORES DIRECCIONALES DEL AGENTE EN P0 (36 DIRECCIONES) ---")
    print(f"Mejor dirección sensor : [{mejor_u[0]:.4f}, {mejor_u[1]:.4f}] (D_u U = {menor_du:.6f})")
    print(f"Dirección -nabla_U/||.||: [{u_teorico[0]:.4f}, {u_teorico[1]:.4f}]\n")

    # 4. Prueba de manejo de excepciones (vector nulo)
    try:
        derivada_direccional(potencial_U, P0, (0.0, 0.0), H_DEFAULT)
    except ValueError as err:
        print(f"--- 4. VALIDACIÓN DE EXCEPCIONES ---\nExcepción capturada correctamente: {err}\n")

    # 5. Sensibilidad del gradiente respecto al paso h (para tabla en LaTeX)
    print("--- 5. SENSIBILIDAD DE nabla_U(P0) AL VARIAR EL PASO h ---")
    for h_val in [1e-1, 1e-3, 1e-5, 1e-7]:
        g_h = gradiente(potencial_U, P0[0], P0[1], h_val)
        print(f"h = {h_val:.0e} -> nabla_U = [{g_h[0]:.10f}, {g_h[1]:.10f}]")
    print()

    # 6. Ejecución de Descenso del Gradiente y primeras 2 iteraciones
    trayectoria, potenciales, motivo = descenso_gradiente(
        potencial_U, P0, alpha=ALPHA, tol=TOL, max_iter=MAX_ITER, h=H_DEFAULT
    )
    print("--- 6. RESULTADOS DEL DESCENSO DEL GRADIENTE ---")
    print(f"Estado de paro         : {motivo}")
    print(f"Iteraciones totales    : {len(trayectoria) - 1}")
    print(f"Punto final alcanzado  : ({trayectoria[-1][0]:.6f}, {trayectoria[-1][1]:.6f})")
    print(f"Potencial final U      : {potenciales[-1]:.6f}")
    if len(trayectoria) >= 3:
        print(f"Iteración 1 (P1)       : ({trayectoria[1][0]:.6f}, {trayectoria[1][1]:.6f}) | U = {potenciales[1]:.6f}")
        print(f"Iteración 2 (P2)       : ({trayectoria[2][0]:.6f}, {trayectoria[2][1]:.6f}) | U = {potenciales[2]:.6f}")
    print("=" * 72)


def main() -> None:
    imprimir_resumen_numerico()

    # Ejecutar múltiples tasas de aprendizaje para la gráfica de convergencia
    resultados_convergencia = {}
    for a in ALPHAS_COMPARACION:
        tray, pots, _ = descenso_gradiente(
            potencial_U, P0, alpha=a, tol=TOL, max_iter=MAX_ITER, h=H_DEFAULT
        )
        resultados_convergencia[a] = (tray, pots)

    tray_principal, _ = resultados_convergencia[ALPHA]
    _, u_prueba = derivada_direccional(potencial_U, P0, V_PRUEBA, H_DEFAULT)

    # Exportación automática de las 3 figuras a 300 dpi (módulo graficas.py)
    try:
        from graficas import generar_todas_las_graficas

        generar_todas_las_graficas(
            f=potencial_U,
            trayectoria=tray_principal,
            p0=np.array(P0, dtype=float),
            meta=np.array(META, dtype=float),
            obstaculos=OBSTACULOS,
            u_prueba=u_prueba,
            resultados_convergencia=resultados_convergencia,
            h=H_DEFAULT,
        )
    except ImportError:
        print("\n[Aviso] Aún no se ha implementado graficas.py; omitiendo exportación de figuras.")


if __name__ == "__main__":
    main()