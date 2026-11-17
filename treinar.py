"""Avaliação em holdout e treinamento final de um modelo de demonstração."""
from src.revisao import exige_revisao
from pathlib import Path
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from src.modelo import carregar_dados, novo_modelo

PASTA_MODELOS = Path(__file__).parent / "modelos"
CAMINHO_MODELO = PASTA_MODELOS / "redetriage.joblib"


def main():
    textos, categorias = carregar_dados()
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        textos, categorias, test_size=0.25, random_state=42, stratify=categorias
    )
    avaliacao = novo_modelo().fit(x_treino, y_treino)
    previstos = avaliacao.predict(x_teste)
    print(f"Treino: {len(x_treino)} registros | Teste: {len(x_teste)} registros")
    print("Resultados apenas ilustrativos: base pequena e sintética.\n")
    print(classification_report(y_teste, previstos, zero_division=0))
    print("Matriz de confusão (ordem alfabética das categorias):")
    print(confusion_matrix(y_teste, previstos, labels=avaliacao.classes_))
    print("Categorias:", ", ".join(avaliacao.classes_))

    # Identifica os chamados classificados incorretamente.
    erros = [
        (texto, categoria_real, categoria_prevista)
        for texto, categoria_real, categoria_prevista
        in zip(x_teste, y_teste, previstos)
        if categoria_real != categoria_prevista
    ]

    print("\n" + "=" * 50)
    print("ANÁLISE DOS ERROS DE CLASSIFICAÇÃO")
    print("=" * 50)

    print(f"Total de erros: {len(erros)} de {len(x_teste)}")

    if not erros:
        print("Nenhum erro encontrado no conjunto de teste.")
    else:
        for numero, (texto, real, prevista) in enumerate(erros, 1):
            print(f"\nErro {numero}:")
            print(f"Descrição: {texto}")
            print(f"Categoria correta: {real}")
            print(f"Categoria prevista: {prevista}")
            print("-" * 50)
    
    # Avaliação da regra de revisão humana
    probabilidades = avaliacao.predict_proba(x_teste)

    enviados_revisao = 0
    erros_automaticos = 0
    acertos_automaticos = 0
    revisoes_corretas = 0
    revisoes_incorretas = 0

    for real, prevista, probs in zip(
        y_teste, previstos, probabilidades
    ):
        ordenadas = sorted(probs, reverse=True)

        maior = ordenadas[0]
        segundo = ordenadas[1]

        precisa_revisao = exige_revisao(probs)

        
        if precisa_revisao:
            enviados_revisao += 1

            if real == prevista:
                revisoes_corretas += 1
            else:
                revisoes_incorretas += 1

        elif real == prevista:
            acertos_automaticos += 1
        else:
            erros_automaticos += 1

    print("\n" + "=" * 50)
    print("AVALIAÇÃO DA REVISÃO HUMANA")
    print("=" * 50)

    print(f"Total de chamados: {len(x_teste)}")
    print(f"Enviados para revisão: {enviados_revisao}")
    print(f"Acertos automáticos: {acertos_automaticos}")
    print(f"Erros automáticos: {erros_automaticos}")

    print("\nDETALHAMENTO DAS REVISÕES")
    print(f"Previsões corretas enviadas para revisão: {revisoes_corretas}")
    print(f"Previsões incorretas enviadas para revisão: {revisoes_incorretas}")

    taxa_revisao = enviados_revisao / len(x_teste)
    taxa_automacao = (
        acertos_automaticos + erros_automaticos
    ) / len(x_teste)

    print(f"Taxa de revisão humana: {taxa_revisao:.1%}")
    print(f"Taxa de automação: {taxa_automacao:.1%}")


    if enviados_revisao > 0:
        print(
            "Atenção: revisão humana não significa "
            "que a previsão estava errada."
        )



    # Modelo final usa todos os registros disponíveis, após a avaliação separada.
    modelo_final = novo_modelo().fit(textos, categorias)
    PASTA_MODELOS.mkdir(exist_ok=True)
    joblib.dump(modelo_final, CAMINHO_MODELO)
    print(f"\nModelo salvo em: {CAMINHO_MODELO}")


if __name__ == "__main__":
    main()
