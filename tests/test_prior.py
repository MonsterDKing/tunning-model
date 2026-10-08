"""Tests del prior armónico (Chordonomicon + Viterbi)."""

import numpy as np

from acordes_gnn.etiquetas import clase, nombre
from acordes_gnn.prior import clase_de_simbolo, viterbi


def test_simbolos_de_chordonomicon():
    pares = {"G": "G", "Amin": "Am", "Fsmin7/A": "F#m", "Csus4": "Csus", "D5": "Dsus",
             "Bmin7b5": "Bdim", "Caug": "Caug", "E7": "E", "Cadd9": "C", "D/Fs": "D"}
    assert {s: nombre(clase_de_simbolo(s)) for s in pares} == pares


def test_viterbi_sin_peso_es_el_maximo_de_cada_paso():
    rng = np.random.default_rng(0)
    emision = rng.dirichlet(np.ones(60), size=20)
    log_cambio = np.log(np.full((60, 60), 1 / 59))
    assert (viterbi(emision, log_cambio, s=0.9, w=0.0) == emision.argmax(1)).all()


def test_el_prior_desempata_hacia_la_progresion_habitual():
    # paso 1 seguro: C. Paso 2 dudoso entre Am (un poco más probable) y G.
    # Si el prior dice que de C casi siempre se va a G, con peso suficiente gana G.
    C, Am, G = clase(0, "mayor"), clase(9, "menor"), clase(7, "mayor")
    emision = np.full((2, 60), 1e-4)
    emision[0, C] = 1
    emision[1, Am], emision[1, G] = 0.52, 0.48
    log_cambio = np.full((60, 60), np.log(1e-3))
    log_cambio[C, G] = np.log(0.9)
    assert viterbi(emision, log_cambio, s=0.5, w=0.0)[1] == Am
    assert viterbi(emision, log_cambio, s=0.5, w=1.0)[1] == G
