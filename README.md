# RedeTriage ML — Triagem de chamados de rede com Machine Learning

Aplicação educacional em **Python** para sugerir a categoria de chamados de suporte de redes a partir de descrições em português. O projeto combina um classificador de texto, uma interface em **Streamlit** e uma regra experimental que encaminha previsões incertas para **revisão humana**.

> **Aviso:** projeto de estudo, com **80 exemplos sintéticos**. Não foi validado com chamados reais e **não deve ser usado como sistema de diagnóstico ou triagem em produção**.

## Funcionalidades

- Sugestão de cinco categorias: **DHCP, DNS, WI-FI, VPN e LENTIDAO**.
- Visualização dos escores de todas as categorias.
- Recomendações iniciais predefinidas por categoria (não geradas por IA).
- Sinalização de previsões incertas para revisão humana.
- Treinamento e avaliação reproduzíveis com divisão de dados definida por `random_state=42`.
- Relatório de classificação, matriz de confusão, listagem de erros e avaliação da taxa de revisão.
- Testes automatizados com `unittest`.

## Como funciona

1. O arquivo `dados/chamados_exemplo.csv` fornece descrições de chamados e seus rótulos.
2. O **TF-IDF de n-gramas de caracteres** transforma textos em representações numéricas.
3. Uma **regressão logística** do scikit-learn aprende a associar essas representações às cinco classes.
4. A interface consulta o classificador e apresenta as pontuações e a categoria sugerida.
5. Uma regra centralizada em `src/revisao.py` recomenda revisão humana quando o maior escore é **menor que 0,30** ou a diferença entre os dois maiores é **menor que 0,07**. Esses limites são experimentais e não foram calibrados para produção.

## Tecnologias

**Python**, **scikit-learn**, **Streamlit**, **joblib** e **unittest**.

## Resultados experimentais

A avaliação usa uma divisão estratificada da base sintética: **60 registros para treinamento** e **20 para teste** (25%), com `random_state=42`. Após a avaliação, o arquivo `treinar.py` treina o modelo final com os **80 registros** e o salva para uso local.

| Indicador | Resultado observado |
| --- | ---: |
| Acurácia no teste | **70%** (14/20) |
| F1-score macro | **0,70** |
| Previsões sinalizadas para revisão | **15/20 (75%)** |
| Previsões liberadas automaticamente | **5/20 (25%)** |
| Erros encaminhados à revisão | **6/6** |
| Previsões corretas encaminhadas à revisão | **9** |
| Erros liberados automaticamente | **0/5** |
| Testes de código (`unittest`) | **9 aprovados** |

**Interpretação:** a regra de revisão reteve os seis erros nessa amostra pequena, mas também reteve nove previsões corretas. Portanto, a automação foi limitada a 25% dos exemplos. Isso **não comprova** ausência de erros futuros; os dados são sintéticos, os escores não foram calibrados e os limites de revisão não foram validados em um conjunto independente.

## Como executar no Windows (CMD)

Na pasta do projeto, com Python instalado:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python treinar.py
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

Abra o endereço indicado pelo Streamlit, normalmente **http://localhost:8501**. Para interromper o aplicativo, pressione **Ctrl+C** no terminal. Para sair do ambiente virtual, execute `deactivate`.

> Se preferir não ativar o ambiente, use diretamente `.venv\Scripts\python.exe` no lugar de `python` nos comandos de instalação, treinamento, testes e execução.

## Estrutura do projeto

```text
redetriage-ml/
├── app.py                     # Interface web com Streamlit
├── treinar.py                 # Treino, avaliação e análise de erros
├── dados/
│   └── chamados_exemplo.csv   # Base sintética
├── src/
│   ├── __init__.py
│   ├── modelo.py              # Pipeline e inferência
│   └── revisao.py             # Regra de revisão humana
├── tests/
│   ├── test_modelo.py
│   └── test_revisao.py
├── requirements.txt
├── .gitignore
└── README.md
```

O arquivo `modelos/redetriage.joblib` é gerado localmente pelo treinamento e **não é versionado**. Para reproduzi-lo, execute `python treinar.py`.

## Limitações e próximos passos

- Base pequena, sintética e com vocabulário limitado; as classes podem se sobrepor em situações reais.
- Resultados de uma única divisão treino/teste não garantem generalização.
- Regra de revisão humana baseada em limiares experimentais, com alta taxa de revisão na amostra atual.
- Sem dados de produção ou integração com sistemas reais de chamados.
- Próximas melhorias possíveis: ampliar exemplos com permissão e anonimização, usar validação cruzada para comparar configurações, calibrar escores, avaliar casos fora do escopo e criar uma demonstração hospedada.

## Contexto do projeto

Projeto de aprendizado e portfólio, desenvolvido e aprimorado com apoio de ferramentas de IA, testes práticos e análise de erros. Seu objetivo é estudar **classificação de texto, avaliação de modelos e integração de ML com uma aplicação web**.
