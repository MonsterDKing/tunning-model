"""Tests del entrenamiento: pesos de la pérdida por acorde."""

import torch

from acordes_gnn.entrenar import pesos_por_acorde
from acordes_gnn.etiquetas import SIN_ETIQUETA as S


def test_cada_acorde_suma_uno():
    # pieza 0: C C C | G | (sin etiqueta) | C   ·   pieza 1: C C (otra pieza: otro tramo aunque sea C)
    y = torch.tensor([0, 0, 0, 35, S, 0, 0, 0])
    pieza = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
    w = pesos_por_acorde(y, pieza)
    assert torch.allclose(w, torch.tensor([1 / 3, 1 / 3, 1 / 3, 1.0, 0.0, 1.0, 0.5, 0.5]))
    assert w.sum() == 4  # 4 acordes con etiqueta → cada uno suma 1
