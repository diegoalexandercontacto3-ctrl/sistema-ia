import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchRun
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes, CommandHandler, ConversationHandler
from datetime import datetime
import pytz

load_dotenv()

ADMIN_ID = 8545681798
CONFIGURANDO = 1

buscador = DuckDuckGoSearchRun()
llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0.7)

class Estado(TypedDict):
    mensaje: str
    tipo: str
    busqueda: str
    respuesta: str
    historial: List
    sistema: str

def esta_abierto():
    zona = pytz.timezone('America/Argentina/Buenos_Aires')
    ahora = datetime.now(zona)
    dia = ahora.weekday()
    hora = ahora.hour

    if dia < 5:  # Lunes a Viernes
        return hora >= 9 and hora < 18, ahora
    elif dia == 5:  # Sabado
        return hora >= 9 and hora < 13, ahora
    else:  # Domingo
        return False, ahora

def get_sistema_base(negocio):
    return f"""Sos DAL, el asistente virtual de {negocio}.
Trabajas para ayudar a los clientes de este negocio.
Informacion del negocio:
- Horarios: Lunes a Viernes 9-18hs, Sabados 9-13hs
- Telefono: +54 11 1234-5678
- Politica de cambios: 30 dias con ticket de compra
Reglas:
- Siempre responde en espanol, de forma amable y profesional
- Si no sabes algo del negocio, di que lo vas a consultar y deja el telefono
- Nunca inventes informacion sobre productos o precios
- Si el cliente esta enojado, primero disculpate y luego ofrece soluciones
- Si el cliente pide hablar con una persona humana o con un responsable, responde exactamente con la palabra: ESCALAR"""

def get_menu():
    return ReplyKeyboardMarkup([
        [KeyboardButton('📋 Horarios'), KeyboardButton('🔄 Politica de cambios')],
        [KeyboardButton('📞 Telefono'), KeyboardButton('💬 Hacer una consulta')],
        [KeyboardButton('🚨 Hablar con una persona')]
    ], resize_keyboard=True)

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

historiales = {}
negocios = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    historiales[chat_id] = []
    negocios[chat_id] = None
    await context.bot.send_photo(
        chat_id=chat_id,
        photo=open('dal_logo.jpg', 'rb'),
        caption='Hola! Soy DAL — Creamos Agentes de IA.\n\nPara personalizar tu experiencia: cual es el nombre de tu negocio?'
    )
    return CONFIGURANDO

async def recibir_negocio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    nombre_negocio = update.message.text
    negocios[chat_id] = nombre_negocio
    await update.message.reply_text(
        f'Perfecto! Desde ahora soy el asistente virtual de {nombre_negocio}.\n\nElegia una opcion o escribime directamente:',
        reply_markup=get_menu()
    )
    return ConversationHandler.END

async def responder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    mensaje = update.message.text
    nombre_cliente = update.effective_user.first_name or 'Cliente'

    if chat_id not in historiales:
        historiales[chat_id] = []

    negocio = negocios.get(chat_id, 'Mi Tienda')
    sistema = get_sistema_base(negocio)
    
    if mensaje == '🚨 Hablar con una persona':
        await update.message.reply_text(
            'Entendido! Voy a avisar a un responsable para que te contacte a la brevedad.',
            reply_markup=get_menu()
        )
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=f'ALERTA ESCALADO\n\nNegocio: {negocio}\nCliente: {nombre_cliente}\nQuiere hablar con una persona.'
        )
        return

    resultado = agente.invoke({
        'mensaje': mensaje,
        'tipo': '',
        'busqueda': '',
        'respuesta': '',
        'historial': historiales[chat_id],
        'sistema': sistema
    })

    respuesta = resultado['respuesta']

    if 'ESCALAR' in respuesta:
        await update.message.reply_text(
            'Entendido! Voy a avisar a un responsable para que te contacte a la brevedad.',
            reply_markup=get_menu()
        )
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=f'ALERTA ESCALADO\n\nNegocio: {negocio}\nCliente: {nombre_cliente}\nMensaje: {mensaje}'
        )
        return

    historiales[chat_id].append(HumanMessage(content=mensaje))
    historiales[chat_id].append(AIMessage(content=respuesta))

    await update.message.reply_text(respuesta, reply_markup=get_menu())

conv_handler = ConversationHandler(
    entry_points=[CommandHandler('start', start)],
    states={CONFIGURANDO: [MessageHandler(filters.TEXT & ~filters.COMMAND, recibir_negocio)]},
    fallbacks=[]
)

app = ApplicationBuilder().token(os.getenv('TELEGRAM_TOKEN')).build()
app.add_handler(conv_handler)
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder))

print('DAL Telegram bot con personalidad configurable iniciado...')
app.run_polling()