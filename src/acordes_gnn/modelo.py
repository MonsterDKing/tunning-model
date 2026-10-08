"""Paso 4 — El modelo: una GNN heterogénea pequeña que clasifica cada nota en un acorde.

Cómo "piensa" una GNN (message passing)
---------------------------------------
1. Cada nodo empieza con su vector de rasgos (18 números para una nota: su clase de altura, su
   altura, duración, posición métrica…). Una capa lineal lo proyecta a ``oculto`` dimensiones.
2. En cada capa, cada nodo RECIBE mensajes de sus vecinos: para cada tipo de arista (``onset``,
   ``consecutive``, ``during``…) se hace la media de los vectores de los vecinos por esa arista y
   se pasa por una matriz aprendida propia de ese tipo (``SAGEConv``). Luego se suman los mensajes
   de todos los tipos y se combinan con el vector del propio nodo.
3. Tras ``capas`` rondas, el vector de una nota resume su vecindario: qué suena con ella, qué hay
   antes y después, qué hay en su pulso y su compás (los nodos virtuales ``beat`` y ``measure``).
4. Una capa lineal final convierte ese vector en 60 puntuaciones (logits), una por acorde
   (12 raíces × 5 calidades). Softmax → probabilidades.

Esto es lo que el heurístico hacía a mano con PESO_ATAQUE, PESO_MELODIA, PESO_BAJO: decidir
cuánto cuenta cada nota según su contexto. Aquí esos "pesos" son las matrices de cada tipo de
arista y se aprenden con descenso de gradiente a partir del cifrado humano.

Tres arquitecturas:
- ``propio`` (por defecto): ``ModeloAcordes``, escrita aquí para que se vea cada pieza.
- ``temporal``: ``ModeloTemporal`` = ``propio`` + una GRU que recorre los ataques en orden (paso 6).
- ``metrical``: ``graphmuse.nn.MetricalGNN`` (la de la librería; SAGE jerárquico + MLP).
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from acordes_gnn.etiquetas import NUM_CLASES
from acordes_gnn.grafo import NUM_RASGOS


class ModeloAcordes(nn.Module):
    def __init__(self, metadata, oculto: int = 64, capas: int = 3, dropout: float = 0.3, clases: int = NUM_CLASES):
        super().__init__()
        tipos_nodo, tipos_arista = metadata
        # 1. proyección de entrada, una por tipo de nodo (una nota y un compás no "significan" igual)
        self.entrada = nn.ModuleDict({t: nn.Linear(NUM_RASGOS, oculto) for t in tipos_nodo})
        # 2. capas de message passing: un SAGEConv por tipo de arista, sumados
        self.convs = nn.ModuleList(
            HeteroConv({et: SAGEConv(oculto, oculto) for et in tipos_arista}, aggr="sum") for _ in range(capas)
        )
        self.normas = nn.ModuleList(nn.ModuleDict({t: nn.LayerNorm(oculto) for t in tipos_nodo}) for _ in range(capas))
        self.dropout = nn.Dropout(dropout)
        # 4. cabeza de clasificación (solo para las notas)
        self.salida = nn.Linear(oculto, clases)

    def codificar(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        """Pasos 1–3: un vector de ``oculto`` números por nota que resume su vecindario."""
        h = {t: self.entrada[t](x).relu() for t, x in x_dict.items()}
        for conv, normas in zip(self.convs, self.normas):
            nuevo = conv(h, edge_index_dict)
            # conexión residual: h + mensaje. Ayuda a que el gradiente llegue a las primeras capas
            h = {t: self.dropout(normas[t](h[t] + nuevo[t].relu())) if t in nuevo else h[t] for t in h}
        return h["note"]

    def forward(self, x_dict: dict, edge_index_dict: dict, onset=None, pieza=None) -> torch.Tensor:
        # onset y pieza solo los usa ModeloTemporal; se aceptan aquí para llamar igual a todos los modelos
        return self.salida(self.codificar(x_dict, edge_index_dict))


def agrupar_por_ataque(onset: torch.Tensor, pieza: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Índice de ataque de cada nota y cuántos ataques tiene cada pieza del lote.

    Un "ataque" es un instante en que empieza al menos una nota. Las notas de la misma pieza con
    el mismo ``onset`` comparten ataque. ``torch.unique`` ordena por pieza y luego por tiempo, así
    que los ataques de cada pieza quedan seguidos y en orden temporal: listos para la GRU.
    """
    claves = torch.stack([pieza.to(onset.dtype), onset], dim=1)
    unicos, ataque = torch.unique(claves, dim=0, return_inverse=True)
    conteos = torch.unique_consecutive(unicos[:, 0], return_counts=True)[1].tolist()
    return ataque, conteos


class ModeloTemporal(ModeloAcordes):
    """Paso 6 — GNN + GRU bidireccional sobre los ataques (la idea de ChordGNN).

    La GNN sola ve unos 3 saltos de grafo (su compás y algo del vecino) y mezcla "antes" y
    "después" en una media. Por eso, cerca de un cambio de acorde, a menudo dice el acorde
    vecino. Aquí, después de la GNN:

    1. se juntan las notas que empiezan a la vez → un vector por ataque (la media de sus notas);
    2. una GRU recorre los ataques EN ORDEN, de izquierda a derecha y de derecha a izquierda.
       Una GRU es una red recurrente: lleva una "memoria" (un vector) que actualiza en cada paso,
       así cada ataque sabe qué vino antes y qué viene después, sin una ventana de tamaño fijo;
    3. cada nota recibe el contexto de su ataque, se pega a su propio vector y se clasifica.
    """

    def __init__(self, metadata, oculto: int = 64, capas: int = 3, dropout: float = 0.3, clases: int = NUM_CLASES):
        super().__init__(metadata, oculto, capas, dropout, clases)
        self.gru = nn.GRU(oculto, oculto, batch_first=True, bidirectional=True)
        # la nota (oculto) + su contexto hacia atrás y hacia delante (2 × oculto)
        self.salida = nn.Linear(3 * oculto, clases)

    def forward(self, x_dict: dict, edge_index_dict: dict, onset=None, pieza=None) -> torch.Tensor:
        h = self.codificar(x_dict, edge_index_dict)
        if pieza is None:  # una sola partitura (evaluación): todas sus notas son de la pieza 0
            pieza = torch.zeros(len(h), dtype=torch.long, device=h.device)
        ataque, conteos = agrupar_por_ataque(onset, pieza)
        # 1. media de las notas de cada ataque
        suma = torch.zeros(sum(conteos), h.size(1), device=h.device).index_add_(0, ataque, h)
        cuantas = torch.zeros(sum(conteos), device=h.device).index_add_(0, ataque, torch.ones(len(h), device=h.device))
        por_ataque = suma / cuantas[:, None]
        # 2. la GRU recorre cada pieza por separado (no debe "recordar" la partitura anterior del lote)
        contexto = torch.cat([self.gru(secuencia[None])[0][0] for secuencia in por_ataque.split(conteos)])
        # 3. de vuelta a las notas: cada una toma el contexto de su ataque
        return self.salida(torch.cat([h, self.dropout(contexto[ataque])], dim=1))


class _Metrical(nn.Module):
    """Adaptador para usar MetricalGNN de GraphMuse con la misma interfaz."""

    def __init__(self, metadata, oculto: int = 64, capas: int = 3, dropout: float = 0.3, clases: int = NUM_CLASES):
        super().__init__()
        from graphmuse.nn import MetricalGNN

        self.red = MetricalGNN(NUM_RASGOS, oculto, clases, capas, metadata, dropout=dropout)

    def forward(self, x_dict: dict, edge_index_dict: dict, onset=None, pieza=None) -> torch.Tensor:
        return self.red(x_dict, edge_index_dict)


ARQUITECTURAS = {"propio": ModeloAcordes, "temporal": ModeloTemporal, "metrical": _Metrical}


def crear_modelo(arquitectura: str, metadata, **kw) -> nn.Module:
    return ARQUITECTURAS[arquitectura](metadata, **kw)


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
