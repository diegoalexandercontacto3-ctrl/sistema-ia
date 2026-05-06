# app.py — Tu agente con interfaz visual
import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
import streamlit as st

load_dotenv()

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="NEXUS — Agente IA",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 NEXUS — Agente Inteligente")
st.caption("Powered by LangGraph + Groq")

# 2. HERRAMIENTAS Y LLM
buscador = DuckDuckGoSearchRun()
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# 3. ESTADO
class Estado(TypedDict):
    mensaje: str
    tipo: str
    busqueda: str
    respuesta: str

# 4. NODOS
def clasificar(estado: Estado):
    prompt = [
        SystemMessage(content="Clasificá el mensaje en UNA sola palabra: queja, busqueda o consulta."),
        HumanMessage(content=estado["mensaje"])
    ]
    resultado = llm.invoke(prompt)
    return {"tipo": resultado.content.strip().lower()}

def buscar_web(estado: Estado):
    resultado = buscador.run(estado["mensaje"])
    return {"busqueda": resultado}

def responder_consulta(estado: Estado):
    prompt = [
        SystemMessage(content="Sos un asistente profesional. Respondé de forma clara y útil en español."),
        HumanMessage(content=estado["mensaje"])
    ]
    resultado = llm.invoke(prompt)
    return {"respuesta": resultado.content}

def responder_con_busqueda(estado: Estado):
    prompt = [
        SystemMessage(content="Respondé usando la información encontrada. Sé claro y conciso. En español."),
        HumanMessage(content=f"""Pregunta: {estado['mensaje']}
        
Información de internet:
{estado['busqueda']}""")
    ]
    resultado = llm.invoke(prompt)
    return {"respuesta": resultado.content}

def manejar_queja(estado: Estado):
    prompt = [
        SystemMessage(content="Sos un agente empático. Disculpate y ofrecé soluciones concretas. En español."),
        HumanMessage(content=estado["mensaje"])
    ]
    resultado = llm.invoke(prompt)
    return {"respuesta": resultado.content}

# 5. DECISIÓN
def decidir(estado: Estado):
    tipo = estado["tipo"]
    if "queja" in tipo:
        return "queja"
    elif "busqueda" in tipo:
        return "buscar"
    else:
        return "consulta"

# 6. GRAFO
@st.cache_resource
def crear_agente():
    grafo = StateGraph(Estado)
    grafo.add_node("clasificar", clasificar)
    grafo.add_node("buscar", buscar_web)
    grafo.add_node("responder_busqueda", responder_con_busqueda)
    grafo.add_node("consulta", responder_consulta)
    grafo.add_node("queja", manejar_queja)
    grafo.set_entry_point("clasificar")
    grafo.add_conditional_edges(
        "clasificar", decidir,
        {"queja": "queja", "buscar": "buscar", "consulta": "consulta"}
    )
    grafo.add_edge("buscar", "responder_busqueda")
    grafo.add_edge("responder_busqueda", END)
    grafo.add_edge("consulta", END)
    grafo.add_edge("queja", END)
    return grafo.compile()

agente = crear_agente()

# 7. HISTORIAL EN PANTALLA
if "historial" not in st.session_state:
    st.session_state.historial = []

for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["rol"]):
        st.write(mensaje["texto"])

# 8. INPUT DEL USUARIO
if entrada := st.chat_input("Escribí tu mensaje..."):
    with st.chat_message("user"):
        st.write(entrada)
    st.session_state.historial.append({"rol": "user", "texto": entrada})

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            resultado = agente.invoke({
                "mensaje": entrada,
                "tipo": "",
                "busqueda": "",
                "respuesta": ""
            })
            respuesta = resultado["respuesta"]
            st.write(respuesta)

    st.session_state.historial.append({"rol": "assistant", "texto": respuesta})