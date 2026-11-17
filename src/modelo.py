"""Treinamento e inferência de categorias de chamados de rede.

Este é um projeto didático: os dados incluídos foram criados para demonstração,
não representam chamados reais de uma central de suporte.
"""
from __future__ import annotations

import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ARQUIVO_DADOS = Path(__file__).resolve().parents[1] / "dados" / "chamados_exemplo.csv"


def carregar_dados(caminho: Path = ARQUIVO_DADOS) -> tuple[list[str], list[str]]:
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    textos = [linha["texto"].strip() for linha in linhas]
    categorias = [linha["categoria"].strip() for linha in linhas]
    if len(textos) < 20 or not all(textos) or not all(categorias):
        raise ValueError("Base insuficiente ou com registros vazios.")
    return textos, categorias


def novo_modelo() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            strip_accents="unicode", lowercase=True
        )),
        ("classificador", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def prever(modelo: Pipeline, texto: str) -> list[tuple[str, float]]:
    if not texto or not texto.strip():
        raise ValueError("Descreva um problema antes de realizar a previsão.")
    probabilidades = modelo.predict_proba([texto])[0]
    categorias = modelo.classes_
    return sorted(zip(categorias, map(float, probabilidades)), key=lambda par: par[1], reverse=True)
