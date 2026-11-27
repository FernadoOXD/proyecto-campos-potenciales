"""
Módulo Matemático: calculo_vectorial.py
Implementación desde cero de derivadas parciales (diferencias finitas centradas),
vector gradiente, derivada direccional, verificación del teorema fundamental
y algoritmo de Descenso del Gradiente para navegación por campos potenciales.
"""

from typing import Callable, Dict, List, Tuple, Union
import numpy as np

Vector2D = Union[Tuple[float, float], List[float], np.ndarray]


def derivada_parcial_x(
    f: Callable[[float, float], float], x: float, y: float, h: float = 1e-5
) -> float:
    """Calcula df/dx mediante diferencias finitas centradas."""
    if h <= 0:
        raise ValueError("El tamaño de paso h debe ser estrictamente positivo.")
    return (f(x + h, y) - f(x - h, y)) / (2.0 * h)


def derivada_parcial_y(
    f: Callable[[float, float], float], x: float, y: float, h: float = 1e-5
) -> float:
    """Calcula df/dy mediante diferencias finitas centradas."""
    if h <= 0:
        raise ValueError("El tamaño de paso h debe ser estrictamente positivo.")
    return (f(x, y + h) - f(x, y - h)) / (2.0 * h)


def gradiente(
    f: Callable[[float, float], float], x: float, y: float, h: float = 1e-5
) -> np.ndarray:
    """Construye el vector gradiente nabla_f(x, y) = [df/dx, df/dy]."""
    df_dx = derivada_parcial_x(f, x, y, h)
    df_dy = derivada_parcial_y(f, x, y, h)
    return np.array([df_dx, df_dy], dtype=float)


def derivada_direccional(
    f: Callable[[float, float], float],
    punto: Vector2D,
    vector_v: Vector2D,
    h: float = 1e-5,
) -> Tuple[float, np.ndarray]:
    """
    Valida que v != 0, normaliza u = v / ||v|| y calcula D_u f = nabla_f · u.

    Retorna:
        (valor_derivada_direccional, vector_unitario_u)
    """
    p = np.asarray(punto, dtype=float)
    v = np.asarray(vector_v, dtype=float)

    if p.shape != (2,) or v.shape != (2,):
        raise ValueError("Tanto el punto como el vector v deben ser bidimensionales (2D).")

    norma_v = float(np.linalg.norm(v))
    if norma_v < 1e-12:
        raise ValueError(
            "Error matemático: el vector dirección v no puede ser el vector nulo (||v|| = 0)."
        )

    u = v / norma_v
    grad = gradiente(f, p[0], p[1], h)
    d_u = float(np.dot(grad, u))
    return d_u, u


def verificar_teorema_direccional(
    f: Callable[[float, float], float],
    punto: Vector2D,
    n_muestras: int = 100,
    h: float = 1e-5,
    semilla: int = 42,
) -> Dict[str, object]:
    """
    Comprueba computacionalmente que -||nabla_f|| <= D_u f <= ||nabla_f||
    evaluando n_muestras vectores dirección aleatorios en el punto dado.
    """
    rng = np.random.default_rng(semilla)
    p = np.asarray(punto, dtype=float)
    grad = gradiente(f, p[0], p[1], h)
    norma_grad = float(np.linalg.norm(grad))

    valores_du: List[float] = []
    cumple_todos = True
    tolerancia_num = 1e-9

    for _ in range(n_muestras):
        v_rand = rng.uniform(-1.0, 1.0, size=2)
        while np.linalg.norm(v_rand) < 1e-8:
            v_rand = rng.uniform(-1.0, 1.0, size=2)

        d_u, _ = derivada_direccional(f, p, v_rand, h)
        valores_du.append(d_u)
        if not (-norma_grad - tolerancia_num <= d_u <= norma_grad + tolerancia_num):
            cumple_todos = False

    return {
        "cumple_teorema": cumple_todos,
        "cota_inferior": -norma_grad,
        "cota_superior": norma_grad,
        "min_observado": min(valores_du),
        "max_observado": max(valores_du),
    }


def evaluar_sensores_agente(
    f: Callable[[float, float], float],
    punto: Vector2D,
    n_direcciones: int = 36,
    h: float = 1e-5,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Simula sensores virtuales en 360 grados alrededor del agente autónomo,
    evalúa D_u f en cada dirección y retorna:
        (mejor_direccion_u, menor_du, direccion_teorica_menos_grad)
    """
    p = np.asarray(punto, dtype=float)
    angulos = np.linspace(0.0, 2.0 * np.pi, n_direcciones, endpoint=False)
    mejor_u = np.zeros(2)
    menor_du = float("inf")

    for theta in angulos:
        v = np.array([np.cos(theta), np.sin(theta)])
        d_u, u = derivada_direccional(f, p, v, h)
        if d_u < menor_du:
            menor_du = d_u
            mejor_u = u

    grad = gradiente(f, p[0], p[1], h)
    norma_g = np.linalg.norm(grad)
    u_teorico = -grad / norma_g if norma_g > 1e-12 else np.zeros(2)

    return mejor_u, menor_du, u_teorico


def descenso_gradiente(
    f: Callable[[float, float], float],
    p0: Vector2D,
    alpha: float = 0.08,
    tol: float = 1e-4,
    max_iter: int = 1000,
    h: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Ejecuta el algoritmo iterativo de Descenso del Gradiente:
        P_{k+1} = P_k - alpha * nabla_f(P_k)

    Retorna:
        (trayectoria_puntos, historial_potencial, motivo_paro)
    """
    if alpha <= 0:
        raise ValueError("La tasa de aprendizaje alpha debe ser estrictamente positiva.")

    p_actual = np.asarray(p0, dtype=float).copy()
    trayectoria: List[np.ndarray] = [p_actual.copy()]
    potenciales: List[float] = [float(f(p_actual[0], p_actual[1]))]
    motivo_paro = f"Máximo de iteraciones alcanzado ({max_iter})"

    for _ in range(max_iter):
        grad = gradiente(f, p_actual[0], p_actual[1], h)
        norma_grad = float(np.linalg.norm(grad))

        if norma_grad < tol:
            motivo_paro = f"Convergencia alcanzada (||nabla U|| < {tol})"
            break

        p_siguiente = p_actual - alpha * grad

        # Protección ante pasos excesivos si se prueba un alpha muy alto en vivo
        if np.linalg.norm(p_siguiente) > 1e5 or np.isnan(p_siguiente).any():
            motivo_paro = "Divergencia numérica detectada (alpha demasiado grande)"
            break

        p_actual = p_siguiente
        trayectoria.append(p_actual.copy())
        potenciales.append(float(f(p_actual[0], p_actual[1])))

    return np.array(trayectoria), np.array(potenciales), motivo_paro