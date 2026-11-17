"""Interface web local para demonstrar inferência com ML."""
from src.revisao import exige_revisao
from pathlib import Path
import joblib
import streamlit as st
from src.modelo import carregar_dados, novo_modelo, prever

CAMINHO_MODELO = Path(__file__).parent / "modelos" / "redetriage.joblib"

ORIENTACOES = {
    "DNS": "Conferir resolução de nomes (nslookup), servidores DNS configurados e conectividade por IP.",
    "WI-FI": "Verificar intensidade do sinal, SSID, interferências e configuração do ponto de acesso.",
    "VPN": "Conferir credenciais, conectividade até o servidor, certificados e logs do cliente VPN.",
    "DHCP": "Verificar atribuição de IP, escopo disponível, gateway e estado do serviço DHCP.",
    "LENTIDAO": "Medir latência, perda, throughput, horário de pico e utilização do enlace.",
}


@st.cache_resource(show_spinner="Preparando o modelo de demonstração...")
def obter_modelo():
    if CAMINHO_MODELO.exists():
        return joblib.load(CAMINHO_MODELO)
    textos, categorias = carregar_dados()
    return novo_modelo().fit(textos, categorias)


st.set_page_config(page_title="RedeTriage ML", page_icon="🛜", layout="centered")
st.title("🛜 RedeTriage ML")
st.caption("Classificação demonstrativa de chamados de rede usando aprendizado de máquina")
st.info(
    "Projeto didático com 80 exemplos sintéticos. Não é um sistema de diagnóstico "
    "confiável ou validado em ambiente real."
)

texto = st.text_area(
    "Descreva o problema de rede",
    placeholder="Ex.: O navegador acusa falha de DNS, mas consigo acessar o servidor pelo IP.",
    height=125,
)
if st.button("Classificar chamado", type="primary"):
    if not texto.strip():
        st.warning("Escreva a descrição de um chamado antes de classificar.")
    else:
        
        ranking = prever(obter_modelo(), texto)

        categoria, escore = ranking[0]
        segundo_escore = ranking[1][1]

        # Limiares experimentais, ainda não calibrados
        precisa_revisao = exige_revisao(
            [pontuacao for _, pontuacao in ranking]
        )

        if precisa_revisao:
            st.warning(
                "Classificação incerta. "
                "Recomendada revisão humana."
            )
            st.write(
                f"Principal hipótese: {categoria} "
                f"(escore: {escore:.1%})"
            )
        else:
            st.success(
                f"Categoria sugerida: {categoria}"
            )
            st.write(
                f"Escore do modelo: {escore:.1%}"
            )
            st.write("**Verificações iniciais:**")
            st.write(ORIENTACOES[categoria])
            st.caption(
                "Orientações fixas, não geradas por IA."
            )

        st.write("**Distribuição das pontuações**")
        for nome, valor in ranking:
            st.write(f"{nome}: {valor:.1%}")
            st.progress(float(valor))


st.divider()
st.caption("Pipeline: TF-IDF de caracteres → Regressão Logística · Python + scikit-learn + Streamlit")
