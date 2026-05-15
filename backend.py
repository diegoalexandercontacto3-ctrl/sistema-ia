import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from flask import Flask, request, jsonify, render_template, session
import secrets

load_dotenv()

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['POST'])
def config():
    data = request.json
    session['negocio'] = data.get('negocio')
    session['historial'] = []
    session['capturando'] = None
    session['lead'] = {}
    return jsonify({'ok': True})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    mensaje = data.get('mensaje')
    negocio = session.get('negocio', 'el negocio')
    historial_raw = session.get('historial', [])
    capturando = session.get('capturando')
    
    historial = []
    for m in historial_raw:
        if m['role'] == 'user':
            historial.append(HumanMessage(content=m['content']))
        else:
            historial.append(AIMessage(content=m['content']))

    if capturando == 'nombre':
        session['lead'] = {'nombre': mensaje}
        session['capturando'] = 'telefono'
        session['historial'] = historial_raw + [
            {'role': 'user', 'content': mensaje},
            {'role': 'assistant', 'content': '¿Cuál es tu número de teléfono?'}
        ]
        return jsonify({'respuesta': '¿Cuál es tu número de teléfono?', 'tipo': 'escalado', 'capturando': 'telefono'})

    if capturando == 'telefono':
        lead = session.get('lead', {})
        lead['telefono'] = mensaje
        session['lead'] = lead
        session['capturando'] = None
        session['historial'] = historial_raw + [
            {'role': 'user', 'content': mensaje},
            {'role': 'assistant', 'content': f"Gracias {lead.get('nombre', '')}. Un responsable te va a contactar a la brevedad al {mensaje}."}
        ]
        return jsonify({'respuesta': f"Gracias {lead.get('nombre', '')}. Un responsable te va a contactar a la brevedad al {mensaje}.", 'tipo': 'escalado_completo', 'capturando': None})

    sistema = get_sistema_base(negocio)
    resultado = agente.invoke({
        'mensaje': mensaje, 'tipo': '', 'busqueda': '', 'respuesta': '',
        'historial': historial, 'sistema': sistema
    })

    respuesta = resultado['respuesta']
    tipo = resultado['tipo']

    if 'ESCALAR' in respuesta:
        session['capturando'] = 'nombre'
        respuesta = '¿Me podés decir tu nombre para avisarle a un responsable?'
        tipo = 'escalado'

    session['historial'] = historial_raw + [
        {'role': 'user', 'content': mensaje},
        {'role': 'assistant', 'content': respuesta}
    ]

    return jsonify({'respuesta': respuesta, 'tipo': tipo, 'capturando': session.get('capturando')})

@app.route('/api/reset', methods=['POST'])
def reset():
    session.clear()
    return jsonify({'ok': True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)