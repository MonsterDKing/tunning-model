"""Tests de los rasgos de nota: grado en la armadura."""

import numpy as np
import torch

from acordes_gnn.grafo import NOMBRES_RASGOS, calcular_rasgos, tonica_de_armadura

GRADO = slice(NOMBRES_RASGOS.index("grado_1"), NOMBRES_RASGOS.index("grado_7") + 1)


def test_tonica_por_circulo_de_quintas():
    # 0 → Do, 2♯ → Re, 1♭ → Fa, 3♭ → Mi♭, 6♯ → Fa♯
    assert tonica_de_armadura(np.array([0, 2, -1, -3, 6])).tolist() == [0, 2, 5, 3, 6]


def test_grado_en_re_mayor():
    # con 2♯ (Re mayor): Fa♯4 (66) es la 3ª y La4 (69) la 5ª
    rasgos = calcular_rasgos(torch.tensor([66, 69]), torch.zeros(2, 5), torch.tensor([2, 2]))
    nombres = [NOMBRES_RASGOS[GRADO][int(i)] for i in rasgos[:, GRADO].argmax(1)]
    assert nombres == ["grado_3", "grado_5"]


def test_transponer_mueve_la_altura_pero_no_el_grado():
    altura, tonica, fijos = torch.tensor([60, 64, 67, 70]), torch.tensor([0, 0, 0, 0]), torch.zeros(4, 5)
    antes = calcular_rasgos(altura, fijos, tonica)
    despues = calcular_rasgos(altura + 5, fijos, (tonica + 5) % 12)
    assert torch.equal(antes[:, GRADO], despues[:, GRADO])
    assert not torch.equal(antes[:, :12], despues[:, :12])
