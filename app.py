import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
import streamlit as st

load_dotenv()

buscador = DuckDuckGoSearchRun()
llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0.7)

def get_sistema_base(negocio):
    return f"""Sos DAL, el asistente virtual de {negocio}. Trabajas para ayudar a los clientes de este negocio.
Informacion del negocio:
- Horarios: Lunes a Viernes 9-18hs, Sabados 9-13hs
- Telefono: +54 11 1234-5678
- Politica de cambios: 30 dias con ticket de compra
Reglas:
- Siempre responde en espanol, de forma amable y profesional
- Si no sabes algo del negocio, di que lo vas a consultar y deja el telefono
- Nunca inventes informacion sobre productos o precios
- Si el cliente esta enojado, primero disculpate y luego ofrece soluciones
- Si el cliente pide hablar con una persona humana, responde exactamente: ESCALAR"""

class Estado(TypedDict):
    mensaje: str
    tipo: str
    busqueda: str
    respuesta: str
    historial: List
    sistema: str

def clasificar(estado):
    prompt = [SystemMessage(content='Clasifica en UNA sola palabra: queja, busqueda o consulta.'),
              HumanMessage(content=estado['mensaje'])]
    resultado = llm.invoke(prompt)
    return {'tipo': resultado.content.strip().lower()}

def buscar_web(estado):
    resultado = buscador.run(estado['mensaje'])
    return {'busqueda': resultado}

def responder_consulta(estado):
    mensajes = [SystemMessage(content=estado['sistema'])]
    mensajes += estado['historial']
    mensajes.append(HumanMessage(content=estado['mensaje']))
    resultado = llm.invoke(mensajes)
    return {'respuesta': resultado.content}

def responder_con_busqueda(estado):
    mensajes = [SystemMessage(content=estado['sistema'])]
    mensajes += estado['historial']
    mensajes.append(HumanMessage(content=f"Pregunta: {estado['mensaje']}\n\nInfo: {estado['busqueda']}"))
    resultado = llm.invoke(mensajes)
    return {'respuesta': resultado.content}

def manejar_queja(estado):
    mensajes = [SystemMessage(content=estado['sistema'] + '\nATENCION: El cliente esta presentando una queja.')]
    mensajes += estado['historial']
    mensajes.append(HumanMessage(content=estado['mensaje']))
    resultado = llm.invoke(mensajes)
    return {'respuesta': resultado.content}

def decidir(estado):
    tipo = estado['tipo']
    if 'queja' in tipo: return 'queja'
    elif 'busqueda' in tipo: return 'buscar'
    else: return 'consulta'

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
agente = grafo.compile()

# ── STREAMLIT UI ──
st.set_page_config(page_title='DAL — Agente de IA', page_icon='🤖')

col1, col2 = st.columns([1, 4])
with col1:
    st.image('dal_logo.jpg', width=80)
with col2:
    st.title('DAL — Creamos Agentes de IA')

if 'negocio' not in st.session_state:
    st.session_state.negocio = None
if 'historial' not in st.session_state:
    st.session_state.historial = []
if 'lead' not in st.session_state:
    st.session_state.lead = {}
if 'capturando' not in st.session_state:
    st.session_state.capturando = None
if 'escalado' not in st.session_state:
    st.session_state.escalado = False
if 'tipo_consulta' not in st.session_state:
    st.session_state.tipo_consulta = None    
if not st.session_state.negocio:
    st.info('Para personalizar el asistente, ingresa el nombre de tu negocio.')
    negocio_input = st.text_input('Nombre del negocio:')
    if st.button('Confirmar') and negocio_input:
        st.session_state.negocio = negocio_input
        st.rerun()
else:
    st.caption(f'Asistente configurado para: {st.session_state.negocio}')

    if st.session_state.tipo_consulta:
        tipo = st.session_state.tipo_consulta
        if 'queja' in tipo:
            st.error('🔴 Queja detectada')
        elif 'busqueda' in tipo:
            st.info('🔵 Búsqueda web')
        else:
            st.success('🟢 Consulta general')

    for msg in st.session_state.historial:
        role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
        with st.chat_message(role):
            st.write(msg.content)

    if st.session_state.escalado:
        st.warning('Un responsable del negocio te va a contactar a la brevedad.')

    if st.session_state.capturando == 'nombre':
        nombre = st.text_input('¿Cuál es tu nombre?')
        if st.button('Enviar nombre') and nombre:
            st.session_state.lead['nombre'] = nombre
            st.session_state.capturando = 'telefono'
            st.rerun()

    elif st.session_state.capturando == 'telefono':
        telefono = st.text_input('¿Cuál es tu número de teléfono?')
        if st.button('Enviar teléfono') and telefono:
            st.session_state.lead['telefono'] = telefono
            st.session_state.capturando = None
            st.session_state.escalado = True
            st.rerun()

    else:
        mensaje = st.chat_input('Escribí tu consulta...')
        if mensaje:
            sistema = get_sistema_base(st.session_state.negocio)
            resultado = agente.invoke({
                'mensaje': mensaje, 'tipo': '', 'busqueda': '', 'respuesta': '',
                'historial': st.session_state.historial, 'sistema': sistema
            })
            respuesta = resultado['respuesta']

            st.session_state.historial.append(HumanMessage(content=mensaje))

            if 'ESCALAR' in respuesta:
                st.session_state.historial.append(AIMessage(content='Entendido, voy a avisar a un responsable para que te contacte.'))
                st.session_state.capturando = 'nombre'
            else:
                st.session_state.tipo_consulta = resultado['tipo']
                st.session_state.historial.append(AIMessage(content=respuesta))

            st.rerun()

    if st.button('Nueva conversación'):
        st.session_state.historial = []
        st.session_state.negocio = None
        st.session_state.lead = {}
        st.session_state.capturando = None
        st.session_state.escalado = False
        st.rerun()    