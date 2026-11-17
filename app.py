"""Interface web local para demonstrar classificação de chamados com ML."""
from pathlib import Path

import joblib
import streamlit as st

from src.modelo import carregar_dados, novo_modelo, prever
from src.revisao import exige_revisao

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


st.set_page_config(
    page_title="RedeTriage ML",
    page_icon="🛜",
    layout="centered",
)

st.title("🛜 RedeTriage ML")
st.caption("Triagem demonstrativa de chamados de rede com Machine Learning")
st.info(
    "Protótipo educacional treinado com 80 exemplos sintéticos. "
    "As previsões não substituem um diagnóstico técnico."
)

with st.container(border=True):
    st.subheader("Descreva o chamado")
    with st.form("formulario_chamado"):
        texto = st.text_area(
            "Qual problema de rede foi observado?",
            placeholder="Ex.: O computador não recebe endereço IP automaticamente.",
            height=110,
        )
        classificar = st.form_submit_button(
            "Classificar chamado", type="primary", use_container_width=True
        )

if classificar:
    if not texto.strip():
        st.session_state.pop("ultima_classificacao", None)
        st.warning("Escreva a descrição de um chamado antes de classificar.")
    else:
        ranking = prever(obter_modelo(), texto)
        st.session_state["ultima_classificacao"] = {
            "texto": texto.strip(),
            "ranking": ranking,
            "precisa_revisao": exige_revisao(
                [pontuacao for _, pontuacao in ranking]
            ),
        }

resultado = st.session_state.get("ultima_classificacao")
if resultado:
    categoria, escore = resultado["ranking"][0]
    precisa_revisao = resultado["precisa_revisao"]

    st.subheader("Resultado da análise")
    st.caption(f"Descrição analisada: {resultado['texto']}")

    with st.container(border=True):
        if precisa_revisao:
            st.warning("Classificação incerta — recomenda-se revisão humana.")
        else:
            st.success("Classificação automática sugerida.")

        col_categoria, col_escore = st.columns(2)
        with col_categoria:
            st.metric(
                "Principal hipótese" if precisa_revisao else "Categoria sugerida",
                categoria,
            )
        with col_escore:
            st.metric("Escore do modelo", f"{escore:.1%}")

        if not precisa_revisao:
            st.markdown("**Verificações iniciais**")
            st.write(ORIENTACOES[categoria])
            st.caption("Orientações fixas; não são geradas pela IA.")
        else:
            st.caption(
                "A hipótese exibida não deve ser tratada como uma classificação confirmada."
            )

    with st.container(border=True):
        st.markdown("**Distribuição dos escores**")
        for nome, valor in resultado["ranking"]:
            col_nome, col_valor = st.columns([4, 1])
            with col_nome:
                st.write(nome)
            with col_valor:
                st.write(f"**{valor:.1%}**")
            st.progress(float(valor))
        st.caption(
            "Escores do classificador não calibrados para decisões operacionais."
        )

st.divider()
st.caption(
    "Pipeline: TF-IDF de caracteres → Regressão Logística · "
    "Python + scikit-learn + Streamlit"
)
