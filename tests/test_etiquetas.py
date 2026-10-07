"""Tests del vocabulario de acordes y de la alineación nota → acorde."""

import numpy as np
import pytest
from music21 import harmony

from acordes_gnn.etiquetas import (
    NUM_CLASES,
    SIN_ETIQUETA,
    AcordeEscrito,
    _desde_simbolo,
    calidad_por_intervalos,
    clase,
    clases_de_altura,
    etiquetar_notas,
    nombre,
    raiz_y_calidad,
    transponer_clase,
    ubicar,
)


def test_vocabulario_tiene_60_clases_y_es_reversible():
    assert NUM_CLASES == 60
    for c in range(NUM_CLASES):
        r, q = raiz_y_calidad(c)
        assert clase(r, q) == c
    assert nombre(0) == "C" and nombre(clase(9, "menor")) == "Am" and nombre(clase(11, "dim")) == "Bdim"


@pytest.mark.parametrize(
    "figura, esperado",
    [
        ("C", "C"), ("C7", "C"), ("Cmaj7", "C"), ("C6", "C"), ("C9", "C"), ("C/E", "C"),
        ("Am", "Am"), ("Am7", "Am"), ("Cm6", "Cm"),
        ("Bdim", "Bdim"), ("Bm7b5", "Bdim"), ("Bdim7", "Bdim"),
        ("C+", "Caug"), ("C7#5", "Caug"),
        ("Csus4", "Csus"), ("Csus2", "Csus"), ("C5", "Csus"), ("C7sus4", "Csus"),
    ],
)  # fmt: skip
def test_calidad_de_cifrados_reales(figura, esperado):
    a = _desde_simbolo(harmony.ChordSymbol(figura), 0, 0.0)
    assert nombre(a.clase) == esperado


def test_calidad_por_intervalos_sin_tercera_es_sus():
    assert calidad_por_intervalos({0, 7}) == "sus"
    assert calidad_por_intervalos({0, 5, 7}) == "sus"


def test_rareza_de_music21_se_conserva_para_la_metrica():
    # music21 dice que C6 es "minor" (lo lee como Am7): la métrica estricta lo hereda de Analisis-AI
    a = _desde_simbolo(harmony.ChordSymbol("C6"), 0, 0.0)
    assert a.calidad == "mayor" and a.calidad_m21 == "minor"


def test_transponer_clase():
    assert nombre(transponer_clase(clase(0, "mayor"), 2)) == "D"
    assert nombre(transponer_clase(clase(9, "menor"), 5)) == "Dm"
    assert nombre(transponer_clase(clase(11, "dim"), -1)) == "Bbdim"
    assert transponer_clase(SIN_ETIQUETA, 3) == SIN_ETIQUETA


def test_clases_de_altura():
    assert clases_de_altura(clase(0, "mayor")) == {0, 4, 7}
    assert clases_de_altura(clase(9, "menor")) == {9, 0, 4}
    assert clases_de_altura(clase(7, "sus")) == {7, 2}


def _acorde(indice, offset, raiz, calidad="mayor"):
    return AcordeEscrito(indice=indice, offset=offset, figura="?", raiz=raiz, calidad=calidad,
                         calidad_m21="major", alturas=frozenset())  # fmt: skip


def test_etiquetar_notas_por_ataque_con_anacrusa_y_vigencia():
    # anacrusa de 1 negra (partitura la pone en t=-1) y luego compases de 4/4
    inicios = np.array([-1.0, 0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0])
    acordes = [_acorde(1, 0.0, 0), _acorde(1, 2.0, 7), _acorde(2, 0.0, 9, "menor")]
    ubicar(acordes, inicios)
    assert [a.tiempo for a in acordes] == [0.0, 2.0, 4.0]
    notas = np.array([-1.0, 0.0, 1.5, 2.0, 3.99, 4.0, 15.5, 20.0])
    y = etiquetar_notas(notas, acordes)
    C, G, Am = clase(0, "mayor"), clase(7, "mayor"), clase(9, "menor")
    # la anacrusa no tiene acorde; Am (compás 3) vale hasta el inicio del compás 7 (t=20)
    assert y.tolist() == [SIN_ETIQUETA, C, C, G, G, Am, Am, SIN_ETIQUETA]


def test_nc_corta_el_acorde():
    inicios = np.array([0.0, 4.0, 8.0])
    acordes = [_acorde(0, 0.0, 0)]
    ubicar(acordes, inicios)
    y = etiquetar_notas(np.array([0.0, 1.0, 2.0, 3.0]), acordes, silencios_en_tiempo=[2.0])
    assert y.tolist() == [0, 0, SIN_ETIQUETA, SIN_ETIQUETA]
