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

Dos arquitecturas:
- ``propio`` (por defecto): ``ModeloAcordes``, escrita aquí para que se vea cada pieza.
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

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        h = {t: self.entrada[t](x).relu() for t, x in x_dict.items()}
        for conv, normas in zip(self.convs, self.normas):
            nuevo = conv(h, edge_index_dict)
            # conexión residual: h + mensaje. Ayuda a que el gradiente llegue a las primeras capas
            h = {t: self.dropout(normas[t](h[t] + nuevo[t].relu())) if t in nuevo else h[t] for t in h}
        return self.salida(h["note"])


class _Metrical(nn.Module):
    """Adaptador para usar MetricalGNN de GraphMuse con la misma interfaz."""

    def __init__(self, metadata, oculto: int = 64, capas: int = 3, dropout: float = 0.3, clases: int = NUM_CLASES):
        super().__init__()
        from graphmuse.nn import MetricalGNN

        self.red = MetricalGNN(NUM_RASGOS, oculto, clases, capas, metadata, dropout=dropout)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        return self.red(x_dict, edge_index_dict)


ARQUITECTURAS = {"propio": ModeloAcordes, "metrical": _Metrical}


def crear_modelo(arquitectura: str, metadata, **kw) -> nn.Module:
    return ARQUITECTURAS[arquitectura](metadata, **kw)


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
