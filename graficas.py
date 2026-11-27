"""
Módulo de Visualización Científica: graficas.py
Generación y exportación automática en alta resolución (300 dpi, PNG y PDF) de:
  - Figura 1 (3D): Superficie z = U(x, y) con la trayectoria de optimización.
  - Figura 2 (2D): Curvas de nivel (contour), campo vectorial (-nabla U),
                   vector dirección u de prueba y pasos del agente autónomo.
  - Figura 3 (Convergencia): Iteraciones vs. valor de U(x_k, y_k) comparando
                             múltiples tasas de aprendizaje (alpha).
"""

import os
from typing import Callable, Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from calculo_vectorial import gradiente

DIRECTORIO_FIGURAS = os.path.join("latex", "figuras")


def _asegurar_directorio() -> None:
    """Crea la carpeta latex/figuras si aún no existe."""
    os.makedirs(DIRECTORIO_FIGURAS, exist_ok=True)


def _guardar_figura(fig: plt.Figure, nombre_base: str) -> None:
    """Exporta una figura de Matplotlib en .png y .pdf a 300 dpi."""
    ruta_png = os.path.join(DIRECTORIO_FIGURAS, f"{nombre_base}.png")
    ruta_pdf = os.path.join(DIRECTORIO_FIGURAS, f"{nombre_base}.pdf")
    fig.savefig(ruta_png, dpi=300, bbox_inches="tight")
    fig.savefig(ruta_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Gráficas] Exportada: {ruta_png} y {ruta_pdf} (300 dpi)")


def graficar_superficie_3d(
    f: Callable[[float, float], float],
    trayectoria: np.ndarray,
    p0: np.ndarray,
    meta: np.ndarray,
) -> None:
    """Figura 1 (3D): Superficie z = U(x, y) con trayectoria trazada sobre el relieve."""
    x_vals = np.linspace(-0.8, 6.0, 140)
    y_vals = np.linspace(-0.8, 6.0, 140)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.vectorize(f)(X, Y)

    z_traj = np.array([f(pt[0], pt[1]) for pt in trayectoria])

    fig = plt.figure(figsize=(9.5, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z, cmap="viridis", alpha=0.82, edgecolor="none", antialiased=True
    )
    fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.08, label="Potencial $U(x, y)$")

    # Trayectoria elevada ligeramente (+0.15) para evitar oclusión con la malla
    ax.plot(
        trayectoria[:, 0],
        trayectoria[:, 1],
        z_traj + 0.15,
        color="crimson",
        linewidth=2.4,
        label="Trayectoria óptima ($-\\nabla U$)",
        zorder=5,
    )

    ax.scatter(
        p0[0], p0[1], f(p0[0], p0[1]) + 0.2,
        color="gold", edgecolor="black", s=80, label="Inicio $P_0$", zorder=6
    )
    ax.scatter(
        meta[0], meta[1], f(meta[0], meta[1]) + 0.2,
        color="lime", edgecolor="black", marker="*", s=150, label="Meta $(x_g, y_g)$", zorder=6
    )

    ax.set_title("Superficie de Potencial Artificial $U(x, y)$ y Trayectoria 3D", fontsize=12, pad=12)
    ax.set_xlabel("Coordenada $x$ (m)", labelpad=8)
    ax.set_ylabel("Coordenada $y$ (m)", labelpad=8)
    ax.set_zlabel("Potencial $U(x, y)$", labelpad=8)
    ax.view_init(elev=32, azim=-128)
    ax.legend(loc="upper left", fontsize=8.5)

    _guardar_figura(fig, "fig1_superficie_3d")


def graficar_contorno_2d(
    f: Callable[[float, float], float],
    trayectoria: np.ndarray,
    p0: np.ndarray,
    meta: np.ndarray,
    obstaculos: List[Dict[str, object]],
    u_prueba: np.ndarray,
    h: float = 1e-5,
) -> None:
    """Figura 2 (2D): Curvas de nivel, campo vectorial -nabla U, vector u y pasos."""
    x_vals = np.linspace(-0.8, 6.0, 220)
    y_vals = np.linspace(-0.8, 6.0, 220)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.vectorize(f)(X, Y)

    fig, ax = plt.subplots(figsize=(8.2, 7.0))

    cf = ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.88)
    ax.contour(X, Y, Z, levels=30, colors="white", linewidths=0.45, alpha=0.45)
    fig.colorbar(cf, ax=ax, label="Nivel de Potencial $U(x, y)$")

    # Campo vectorial normalizado de descenso (-nabla U)
    xq = np.linspace(-0.5, 5.8, 19)
    yq = np.linspace(-0.5, 5.8, 19)
    XQ, YQ = np.meshgrid(xq, yq)
    U_vec = np.zeros_like(XQ)
    V_vec = np.zeros_like(YQ)

    for i in range(XQ.shape[0]):
        for j in range(XQ.shape[1]):
            g = gradiente(f, XQ[i, j], YQ[i, j], h=h)
            norm = np.linalg.norm(g)
            if norm > 1e-9:
                U_vec[i, j] = -g[0] / norm
                V_vec[i, j] = -g[1] / norm

    ax.quiver(
            XQ, YQ, U_vec, V_vec,
            color="white", alpha=0.65, scale=28, width=0.003,
            label=r"Campo direccional $-\nabla U / \|\nabla U\|$"
        )

    # Pasos del algoritmo de Descenso del Gradiente
    ax.plot(
        trayectoria[:, 0],
        trayectoria[:, 1],
        color="crimson",
        linewidth=2.0,
        marker="o",
        markersize=3.2,
        label="Pasos del agente ($P_k$)",
        zorder=4,
    )

    # Vector unitario u de prueba dibujado desde P0
    ax.quiver(
        p0[0], p0[1], u_prueba[0], u_prueba[1],
        angles="xy", scale_units="xy", scale=0.85,
        color="gold", width=0.009, zorder=6,
        label=f"Vector prueba $u = [{u_prueba[0]:.2f}, {u_prueba[1]:.2f}]$"
    )

    # Marcadores de inicio, meta y obstáculos
    ax.scatter(p0[0], p0[1], color="gold", edgecolor="black", s=90, zorder=7, label="Inicio $P_0$")
    ax.scatter(meta[0], meta[1], color="lime", edgecolor="black", marker="*", s=180, zorder=7, label="Meta $(5, 5)$")

    for idx, obs in enumerate(obstaculos, start=1):
        xo, yo = obs["centro"]
        ax.scatter(
            xo, yo, color="black", edgecolor="white", marker="X", s=110, zorder=7,
            label="Obstáculos (máx. locales)" if idx == 1 else ""
        )

    ax.set_title("Mapa de Curvas de Nivel, Campo Vectorial $-\\nabla U$ y Trayectoria 2D", fontsize=11.5)
    ax.set_xlabel("Coordenada $x$ (m)")
    ax.set_ylabel("Coordenada $y$ (m)")
    ax.set_xlim(-0.8, 6.0)
    ax.set_ylim(-0.8, 6.0)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8.0, framealpha=0.9)

    _guardar_figura(fig, "fig2_contorno_2d")


def graficar_convergencia(
    resultados_convergencia: Dict[float, Tuple[np.ndarray, np.ndarray]]
) -> None:
    """Figura 3 (Convergencia): Iteraciones vs. potencial U(x_k, y_k) para distintas alpha."""
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    estilos = ["-", "--", "-."]
    colores = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    for idx, (alpha, (_, potenciales)) in enumerate(resultados_convergencia.items()):
        iteraciones = np.arange(len(potenciales))
        estilo = estilos[idx % len(estilos)]
        color = colores[idx % len(colores)]
        ax.plot(
            iteraciones,
            potenciales,
            linestyle=estilo,
            color=color,
            linewidth=2.0,
            label=f"$\\alpha = {alpha}$ ({len(potenciales) - 1} iteraciones)",
        )

    ax.set_title("Convergencia del Descenso del Gradiente para Distintas Tasas $\\alpha$", fontsize=11.5)
    ax.set_xlabel("Número de iteración ($k$)")
    ax.set_ylabel("Valor de la función potencial $U(x_k, y_k)$")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper right", fontsize=9.5)

    _guardar_figura(fig, "fig3_convergencia")


def generar_todas_las_graficas(
    f: Callable[[float, float], float],
    trayectoria: np.ndarray,
    p0: np.ndarray,
    meta: np.ndarray,
    obstaculos: List[Dict[str, object]],
    u_prueba: np.ndarray,
    resultados_convergencia: Dict[float, Tuple[np.ndarray, np.ndarray]],
    h: float = 1e-5,
) -> None:
    """Orquesta y exporta las 3 figuras requeridas a 300 dpi."""
    _asegurar_directorio()
    graficar_superficie_3d(f, trayectoria, p0, meta)
    graficar_contorno_2d(f, trayectoria, p0, meta, obstaculos, u_prueba, h=h)
    graficar_convergencia(resultados_convergencia)