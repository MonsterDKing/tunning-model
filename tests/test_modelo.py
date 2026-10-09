"""Tests del modelo temporal: agrupación por ataques y alcance de la GRU."""

import torch

from acordes_gnn.etiquetas import NUM_CLASES
from acordes_gnn.grafo import NUM_RASGOS
from acordes_gnn.modelo import ModeloAcordes, ModeloTemporal, agrupar_por_ataque

METADATA = (["note"], [("note", "onset", "note")])


def test_agrupar_por_ataque():
    # pieza 0: notas en t=0, 0, 1 → 2 ataques · pieza 1: notas en t=2, 0 → 2 ataques, ordenados en el tiempo
    onset = torch.tensor([0.0, 0.0, 1.0, 2.0, 0.0])
    pieza = torch.tensor([0, 0, 0, 1, 1])
    ataque, conteos = agrupar_por_ataque(onset, pieza)
    assert ataque.tolist() == [0, 0, 1, 3, 2]
    assert conteos == [2, 2]


def _salida_primera_nota(modelo, x, onset):
    sin_aristas = {("note", "onset", "note"): torch.empty(2, 0, dtype=torch.long)}
    with torch.no_grad():
        return modelo({"note": x}, sin_aristas, onset=onset)[0]


def test_la_gru_ve_notas_lejanas_y_la_gnn_sola_no():
    """Sin aristas, la GNN no puede saber nada de la última nota; la GRU sí, por el orden temporal."""
    torch.manual_seed(0)
    x = torch.rand(5, NUM_RASGOS)
    onset = torch.arange(5, dtype=torch.float)
    otra = x.clone()
    otra[4] = torch.rand(NUM_RASGOS)  # cambiamos solo la última nota
    for clase, debe_cambiar in ((ModeloAcordes, False), (ModeloTemporal, True)):
        modelo = clase(METADATA).eval()
        antes, despues = _salida_primera_nota(modelo, x, onset), _salida_primera_nota(modelo, otra, onset)
        assert antes.shape == (NUM_CLASES,)
        assert (not torch.allclose(antes, despues)) == debe_cambiar, clase.__name__


def test_cabeza_factorizada_es_una_distribucion_sobre_60_clases():
    """Exp. 25: log p(raíz) + log p(calidad) → 60 logits cuyas probabilidades suman 1 y cuya entropía
    cruzada es la suma de las de raíz y calidad."""
    from acordes_gnn.etiquetas import clase
    from acordes_gnn.modelo import CabezaFactorizada

    torch.manual_seed(0)
    cabeza = CabezaFactorizada(16)
    h = torch.rand(4, 16)
    logits = cabeza(h)
    assert logits.shape == (4, NUM_CLASES)
    assert torch.allclose(logits.exp().sum(-1), torch.ones(4), atol=1e-5)
    y = torch.tensor([clase(0, "mayor"), clase(9, "menor"), clase(11, "dim"), clase(7, "sus")])
    conjunta = torch.nn.functional.cross_entropy(logits, y)
    por_partes = torch.nn.functional.cross_entropy(cabeza.raiz(h), y // 5) + torch.nn.functional.cross_entropy(cabeza.calidad(h), y % 5)
    assert torch.allclose(conjunta, por_partes, atol=1e-5)
