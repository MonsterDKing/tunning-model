"""Tests del entrenamiento: pesos de la pérdida por acorde."""

import torch

from acordes_gnn.entrenar import pesos_por_acorde, pesos_por_calidad
from acordes_gnn.etiquetas import SIN_ETIQUETA as S


def test_cada_acorde_suma_uno():
    # pieza 0: C C C | G | (sin etiqueta) | C   ·   pieza 1: C C (otra pieza: otro tramo aunque sea C)
    y = torch.tensor([0, 0, 0, 35, S, 0, 0, 0])
    pieza = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
    w = pesos_por_acorde(y, pieza)
    assert torch.allclose(w, torch.tensor([1 / 3, 1 / 3, 1 / 3, 1.0, 0.0, 1.0, 0.5, 0.5]))
    assert w.sum() == 4  # 4 acordes con etiqueta → cada uno suma 1


class _Nota:  # imita grafo["note"] con solo la etiqueta
    def __init__(self, y):
        self.y = y


def test_peso_por_calidad_premia_lo_raro():
    # 3 acordes C (mayor) y 1 acorde Cm (menor): con alfa 1 el menor pesa 3 veces el mayor
    piezas = [{"grafo": {"note": _Nota(torch.tensor([0, 0, 35, 35, 0, 1, 1]))}}]
    w = pesos_por_calidad(piezas, 1.0)
    assert w[0] == 1 and w[1] == 3
    assert torch.allclose(pesos_por_calidad(piezas, 0.0), torch.ones(5))
