"""Tests de la métrica (debe comportarse como validar() de Analisis-AI)."""

import torch
from music21 import harmony

from acordes_gnn.etiquetas import NUM_CLASES, _desde_simbolo, clase
from acordes_gnn.evaluar import comparar, notas_del_instante, predecir_en_puntos


def ref(figura):
    a = _desde_simbolo(harmony.ChordSymbol(figura), 0, 0.0)
    return {"raiz": a.raiz, "calidad_m21": a.calidad_m21, "alturas": a.alturas}


def test_acierto_exacto():
    assert comparar(clase(0, "mayor"), ref("C")) == (True, True, True)
    assert comparar(clase(9, "menor"), ref("Am7")) == (True, True, True)


def test_septima_cuenta_como_su_triada():
    assert comparar(clase(7, "mayor"), ref("G7")) == (True, True, True)
    assert comparar(clase(11, "dim"), ref("Bm7b5")) == (True, True, True)


def test_raiz_bien_calidad_mal():
    assert comparar(clase(0, "menor"), ref("C")) == (True, False, False)


def test_equivalente_por_inclusion():
    # Am ⊂ Fmaj7: raíz mal, pero mismas notas → equivalente
    assert comparar(clase(9, "menor"), ref("Fmaj7")) == (False, False, True)
    # Csus = {C, G} ⊂ C → equivalente, aunque la calidad estricta ("other" vs "major") falle
    assert comparar(clase(0, "sus"), ref("C")) == (True, False, True)
    # sus4 escrito y sus predicho: music21 dice "other" y "sus" → "other": acierto estricto
    assert comparar(clase(0, "sus"), ref("Csus4")) == (True, True, True)


def test_rareza_c6():
    # music21 llama "minor" a C6 → predecir C mayor falla la estricta, pero es equivalente
    assert comparar(clase(0, "mayor"), ref("C6")) == (True, False, True)


def test_notas_del_instante_prefiere_ataques():
    onset = torch.tensor([0.0, 0.0, 1.0, 2.0])
    dur = torch.tensor([4.0, 1.0, 1.0, 1.0])
    assert notas_del_instante(onset, dur, 1.0).tolist() == [2]  # ataca en 1 (la redonda no vota)
    assert notas_del_instante(onset, dur, 1.5).tolist() == [0, 2]  # nadie ataca: lo que suena
    onset2, dur2 = torch.tensor([0.0, 3.0]), torch.tensor([1.0, 1.0])
    assert notas_del_instante(onset2, dur2, 2.0).tolist() == [1]  # silencio: el siguiente ataque


def test_prediccion_por_promedio_de_probabilidades():
    prob = torch.zeros(3, NUM_CLASES)
    G, C = clase(7, "mayor"), clase(0, "mayor")
    prob[0, G], prob[0, C] = 0.4, 0.6  # duda
    prob[1, G], prob[1, C] = 0.4, 0.6
    prob[2, G], prob[2, C] = 0.99, 0.01  # segura
    onset, dur = torch.zeros(3), torch.ones(3)
    assert predecir_en_puntos(prob, onset, dur, [0.0]) == [G]  # media: G 0.60, C 0.40


def test_ventana_hacia_delante_vota_con_menos_peso():
    """Exp. 24: las notas que atacan dentro de las 2 negras siguientes votan con peso 0.3; fuera, no."""
    C, G = 0, 7 * 5
    onset = torch.tensor([0.0, 1.0, 3.0])
    dur = torch.ones(3)
    prob = torch.zeros(3, NUM_CLASES)
    prob[0, C], prob[1, G], prob[2, G] = 0.6, 1.0, 1.0  # en t=0 solo vota la primera (C) salvo por la ventana
    prob[0, G] = 0.4
    assert predecir_en_puntos(prob, onset, dur, [0.0], ventana=0) == [C]  # sin ventana: C 0.6 > G 0.4
    assert predecir_en_puntos(prob, onset, dur, [0.0]) == [G]  # con la nota de t=1 (peso 0.3): G 0.7 > C 0.6
    assert predecir_en_puntos(prob, onset, dur, [0.0], ventana=0.5) == [C]  # t=1 queda fuera de media negra
