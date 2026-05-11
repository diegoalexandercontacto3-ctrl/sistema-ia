import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes, CommandHandler

load_dotenv()

NEGOCIO = {
    'nombre': 'Mi Tienda Demo',
    'rubro': 'comercio minorista',
    'horarios': 'Lunes a Viernes 9-18hs, Sabados 9-13hs',
    'telefono': '+54 11 1234-5678',
    'politica_cambios': 'Cambios dentro de los 30 dias con ticket de compra',
    'agente_nombre': 'NEXUS'
}

SISTEMA_BASE = f"""Sos {NEGOCIO['agente_nombre']}, el asistente virtual de {NEGOCIO['nombre']}.
Trabajas para ayudar a los clientes de este {NEGOCIO['rubro']}.
Informacion del negocio:
- Horarios: {NEGOCIO['horarios']}
- Telefono: {NEGOCIO['telefono']}
- Politica de cambios: {NEGOCIO['politica_cambios']}
Reglas:
- Siempre responde en espanol, de forma amable y profesional
- Si no sabes algo del negocio, di que lo vas a consultar y deja el telefono
- Nunca inventes informacion sobre productos o precios
- Si el cliente esta enojado, primero disculpate y luego ofrece soluciones"""

buscador = DuckDuckGoSearchRun()
llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0.7)

class Estado(TypedDict):
    mensaje: str
    tipo: str
    busqueda: str
    respuesta: str
    historial: List

def clasificar(estado):
    prompt = [SystemMessage(content='Clasifica en UNA sola palabra: queja, busqueda o consulta.'),
              HumanMessage(content=estado['mensaje'])]
    resultado = llm.invoke(prompt)
    return {'tipo': resultado.content.strip().lower()}

def buscar_web(estado):
    resultado = buscador.run(estado['mensaje'])
    return {'busqueda': resultado}

def responder_consulta(estado):
    mensajes = [SystemMessage(content=SISTEMA_BASE)]
    mensajes += estado['historial']
    mensajes.append(HumanMessage(content=estado['mensaje']))
    resultado = llm.invoke(mensajes)
    return {'respuesta': resultado.content}

def responder_con_busqueda(estado):
    mensajes = [SystemMessage(content=SISTEMA_BASE)]
    mensajes += estado['historial']
    mensajes.append(HumanMessage(content=f"Pregunta: {estado['mensaje']}\n\nInfo: {estado['busqueda']}"))
    resultado = llm.invoke(mensajes)
    return {'respuesta': resultado.content}

def manejar_queja(estado):
    mensajes = [SystemMessage(content=SISTEMA_BASE + '\nATENCION: El cliente esta presentando una queja.')]
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

historiales = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    historiales[update.effective_chat.id] = []
    await update.message.reply_text('Hola! Soy NEXUS, el asistente virtual de Mi Tienda Demo. Como puedo ayudarte?')

async def responder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    mensaje = update.message.text

    if chat_id not in historiales:
        historiales[chat_id] = []
    
    resultado = agente.invoke({
        'mensaje': mensaje,
        'tipo': '',
        'busqueda': '',
        'respuesta': '',
        'historial': historiales[chat_id]
    })

    respuesta = resultado['respuesta']
    historiales[chat_id].append(HumanMessage(content=mensaje))
    historiales[chat_id].append(AIMessage(content=respuesta))

    await update.message.reply_text(respuesta)

app = ApplicationBuilder().token(os.getenv('TELEGRAM_TOKEN')).build()
app.add_handler(CommandHandler('start', start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder))

print('NEXUS Telegram bot iniciado...')
app.run_polling()    