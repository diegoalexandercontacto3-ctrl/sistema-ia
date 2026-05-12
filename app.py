import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
import streamlit as st

load_dotenv()

NEGOCIO = {
    'nombre': 'DAL — Creamos Agentes de IA',
    'rubro': 'desarrollo de agentes de inteligencia artificial',
    'horarios': 'Lunes a Viernes 9-18hs',
    'telefono': '+54 11 1234-5678',
    'politica_cambios': 'Revisamos cada proyecto y ofrecemos soporte post-entrega',
    'agente_nombre': 'DAL'
}

SISTEMA_BASE = f"""Sos {NEGOCIO['agente_nombre']}, el asistente virtual de {NEGOCIO['nombre']}.
Trabajás para ayudar a los clientes de este {NEGOCIO['rubro']}.

Información del negocio:
- Horarios: {NEGOCIO['horarios']}
- Teléfono: {NEGOCIO['telefono']}
- Política de cambios: {NEGOCIO['politica_cambios']}

Reglas:
- Siempre respondé en español, de forma amable y profesional
- Si no sabés algo del negocio, decí que lo vas a consultar y dejá el teléfono
- Nunca inventés información sobre productos o precios
- Si el cliente está enojado, primero disculpate y luego ofrecé soluciones"""

st.set_page_config(page_title="DAL — Creamos Agentes de IA", page_icon="🤖", layout='centered')
st.image('dal_logo.jpg', width=300)
st.title("DAL — Creamos Agentes de IA")
st.caption(f"Asistente virtual especializado | {NEGOCIO['horarios']}")

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
    mensajes = [SystemMessage(content=SISTEMA_BASE)]
    mensajes += estado['historial']
    mensajes.append(HumanMessage(content=estado['mensaje']))
    resultado = llm.invoke(mensajes)
    return {'respuesta': resultado.content}

def responder_con_busqueda(estado: Estado):
    mensajes = [SystemMessage(content=SISTEMA_BASE)]
    mensajes += estado['historial']
    mensajes.append(HumanMessage(content=f"Pregunta: {estado['mensaje']}\n\nInfo adicional: {estado['busqueda']}"))
    resultado = llm.invoke(mensajes)
    return {'respuesta': resultado.content}

def manejar_queja(estado: Estado):
    mensajes = [SystemMessage(content=SISTEMA_BASE + "\nATENCIÓN: El cliente está presentando una queja. Priorizá la empatía y la solución.")]
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

if entrada := st.chat_input('¿En qué puedo ayudarte?'):
    with st.chat_message('user'):
        st.write(entrada)
    st.session_state.historial_visual.append({'rol': 'user', 'texto': entrada})

    with st.chat_message('assistant'):
        with st.spinner('🧠 Analizando tu consulta...'):
            resultado = agente.invoke({
                'mensaje': entrada,
                'tipo': '',
                'busqueda': '',
                'respuesta': '',
                'historial': st.session_state.historial_llm
            })
        respuesta = resultado['respuesta']
        tipo = resultado['tipo']
        st.write(respuesta)

        if 'queja' in tipo:
            st.caption("⚠️ Queja detectada — respuesta empática activada")
        elif 'busqueda' in tipo:
            st.caption("🔍 Búsqueda web realizada")
        else:
            st.caption("💬 Consulta respondida")

    st.session_state.historial_visual.append({'rol': 'assistant', 'texto': respuesta})
    st.session_state.historial_llm.append(HumanMessage(content=entrada))
    st.session_state.historial_llm.append(AIMessage(content=respuesta))