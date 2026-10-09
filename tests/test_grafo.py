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
    rasgos = calcular_rasgos(torch.tensor([66, 69]), torch.zeros(2, 6), torch.tensor([2, 2]))
    nombres = [NOMBRES_RASGOS[GRADO][int(i)] for i in rasgos[:, GRADO].argmax(1)]
    assert nombres == ["grado_3", "grado_5"]


def test_transponer_mueve_la_altura_pero_no_el_grado():
    altura, tonica, fijos = torch.tensor([60, 64, 67, 70]), torch.tensor([0, 0, 0, 0]), torch.zeros(4, 6)
    antes = calcular_rasgos(altura, fijos, tonica)
    despues = calcular_rasgos(altura + 5, fijos, (tonica + 5) % 12)
    assert torch.equal(antes[:, GRADO], despues[:, GRADO])
    assert not torch.equal(antes[:, :12], despues[:, :12])


def test_es_bajo_cuenta_las_notas_que_siguen_sonando():
    import numpy as np
    from acordes_gnn.grafo import es_bajo

    na = np.array([(0.0, 4.0, 48), (0.0, 1.0, 64), (1.0, 1.0, 67), (4.0, 1.0, 60)],
                  dtype=[("onset_quarter", "f4"), ("duration_quarter", "f4"), ("pitch", "i4")])
    # Do3 (largo) es el bajo; el Sol4 de la negra 1 no lo es porque el Do3 sigue sonando; el Do4 final sí
    assert es_bajo(na).tolist() == [1, 0, 0, 1]


def test_mutar_calidad_mueve_la_nota_y_la_etiqueta_juntas():
    """Exp. 26: Do-Mi-Sol escrito como Do mayor → Do menor (Mi → Mi♭) o Do aumentado (Sol → Sol♯)."""
    import torch
    from torch_geometric.data import HeteroData

    from acordes_gnn.dataset import mutar_calidad
    from acordes_gnn.etiquetas import clase

    g = HeteroData()
    g["note"].pitch = torch.tensor([60, 64, 67, 62])  # Do4 Mi4 Sol4 + un Re4 sin etiqueta (no se toca)
    g["note"].y = torch.tensor([clase(0, "mayor")] * 3 + [-100])
    g["note"].fijos = torch.zeros(4, 6)
    g["note"].tonica = torch.zeros(4, dtype=torch.long)
    esperado = {clase(0, "menor"): [60, 63, 67], clase(0, "aum"): [60, 64, 68], clase(0, "sus"): [60, 65, 67]}
    vistos = set()
    for conjunto in ("todas", "raras"):
        for semilla in range(6):
            h = g.clone()
            mutar_calidad(h, fraccion=1.0, semilla=semilla, conjunto=conjunto)
            assert h["note"].y[3] == -100 and h["note"].pitch[3] == 62
            nueva = int(h["note"].y[0])
            assert h["note"].y[:3].tolist() == [nueva] * 3  # todo el acorde cambia de etiqueta a la vez
            assert h["note"].pitch[:3].tolist() == esperado[nueva]
            assert h["note"].x[1, :12].argmax() == h["note"].pitch[1] % 12  # los rasgos se recalculan
            vistos.add(nueva)
    assert vistos == set(esperado)  # entre los dos conjuntos salen las tres calidades
