
"""Regras experimentais para encaminhamento à revisão humana."""


def exige_revisao(pontuacoes, limite=0.30, margem_minima=0.07):
    """
    Decide se uma previsão precisa de revisão humana.

    pontuacoes: probabilidades das categorias.
    limite: pontuação mínima para classificação automática.
    margem_minima: diferença mínima entre as duas maiores pontuações.
    """

    ordenadas = sorted(
        [float(valor) for valor in pontuacoes],
        reverse=True
    )

    if len(ordenadas) < 2:
        raise ValueError("São necessárias pelo menos duas categorias.")

    maior = ordenadas[0]
    segunda = ordenadas[1]

    return (
        maior < limite
        or (maior - segunda) < margem_minima
    )
