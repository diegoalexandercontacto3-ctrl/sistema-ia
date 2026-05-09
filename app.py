# app.py — NEXUS con memoria de conversación
import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
import streamlit as st

load_dotenv()

st.set_page_config(page_title="NEXUS — Agente IA", page_icon="🤖", layout="centered")
st.title("🤖 NEXUS — Agente Inteligente")
st.caption("Powered by LangGraph + Groq | Con memoria de conversación")

col1, col2 = st.columns([6, 1])
with col1:
    if len(st.session_state.get('historial_visual', [])) > 0:
        total = len(st.session_state.historial_visual) // 2
        st.caption(f"💬 {total} mensaje{'s' if total != 1 else ''}")
with col2:
    if st.button("🗑️ Nueva"):
        st.session_state.historial_visual = []
        st.session_state.historial_llm = []
        st.rerun()

buscador = DuckDuckGoSearchRun()
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

class Estado(TypedDict):
    mensaje: str
    tipo: str
    busqueda: str
    respuesta: str
    historial: List

def clasificar(estado: Estado):
    prompt = [SystemMessage(content="Clasificá en UNA sola palabra: queja, busqueda o consulta."),
              HumanMessage(content=estado['mensaje'])]
    resultado = llm.invoke(prompt)
    return {'tipo': resultado.content.strip().lower()}

def buscar_web(estado: Estado):
    resultado = buscador.run(estado['mensaje'])
    return {'busqueda': resultado}

def responder_consulta(estado: Estado):
    mensajes = [SystemMessage(content="Sos un asistente profesional. Respondé en español. Recordás toda la conversación anterior.")]
    mensajes += estado['historial']
    mensajes.append(HumanMessage(content=estado['mensaje']))
    resultado = llm.invoke(mensajes)
    return {'respuesta': resultado.content}

def responder_con_busqueda(estado: Estado):
    mensajes = [SystemMessage(content="Respondé usando la información encontrada. En español.")]
    mensajes += estado['historial']
    mensajes.append(HumanMessage(content=f"Pregunta: {estado['mensaje']}\n\nInfo: {estado['busqueda']}"))
    resultado = llm.invoke(mensajes)
    return {'respuesta': resultado.content}

def manejar_queja(estado: Estado):
    mensajes = [SystemMessage(content="Sos un agente empático. Disculpate y ofrecé soluciones. En español.")]
    mensajes += estado['historial']
    mensajes.append(HumanMessage(content=estado['mensaje']))
    resultado = llm.invoke(mensajes)
    return {'respuesta': resultado.content}

def decidir(estado: Estado):
    tipo = estado['tipo']
    if 'queja' in tipo: return 'queja'
    elif 'busqueda' in tipo: return 'buscar'
    else: return 'consulta'

@st.cache_resource
def crear_agente():
    grafo = StateGraph(Estado)
    grafo.add_node('clasificar', clasificar)
    grafo.add_node('buscar', buscar_web)
    grafo.add_node('responder_busqueda', responder_con_busqueda)
    grafo.add_node('consulta', responder_consulta)
    grafo.add_node('queja', manejar_queja)
    grafo.set_entry_point('clasificar')
    grafo.add_conditional_edges('clasificar', decidir,
        {'queja': 'queja', 'buscar': 'buscar', 'consulta': 'consulta'})
    grafo.add_edge('buscar', 'responder_busqueda')
    grafo.add_edge('responder_busqueda', END)
    grafo.add_edge('consulta', END)
    grafo.add_edge('queja', END)
    return grafo.compile()

agente = crear_agente()

if 'historial_visual' not in st.session_state:
    st.session_state.historial_visual = []

if 'historial_llm' not in st.session_state:
    st.session_state.historial_llm = []

for mensaje in st.session_state.historial_visual:
    with st.chat_message(mensaje['rol']):
        st.write(mensaje['texto'])

if entrada := st.chat_input('Escribí tu mensaje...'):
    with st.chat_message('user'):
        st.write(entrada)
    st.session_state.historial_visual.append({'rol': 'user', 'texto': entrada})

    with st.chat_message('assistant'):
        with st.spinner('Pensando...'):
            resultado = agente.invoke({
                'mensaje': entrada,
                'tipo': '',
                'busqueda': '',
                'respuesta': '',
                'historial': st.session_state.historial_llm
            })
            respuesta = resultado['respuesta']
            st.write(respuesta)
            
    st.session_state.historial_visual.append({'rol': 'assistant', 'texto': respuesta})
    st.session_state.historial_llm.append(HumanMessage(content=entrada))
    st.session_state.historial_llm.append(AIMessage(content=respuesta)) 