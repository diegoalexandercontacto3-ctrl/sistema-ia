# maquina.py — Agente con herramienta de búsqueda web
import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# 1. HERRAMIENTA DE BÚSQUEDA
buscador = DuckDuckGoSearchRun()

# 2. ESTADO
class Estado(TypedDict):
    pregunta: str
    busqueda: str
    respuesta: str

# 3. LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# 4. NODOS
def buscar_en_web(estado: Estado):
    print("[Nodo 1] Buscando en internet...")
    resultado = buscador.run(estado["pregunta"])
    print("[Nodo 1] Búsqueda completada")
    return {"busqueda": resultado}

def generar_respuesta(estado: Estado):
    print("[Nodo 2] Generando respuesta...")
    prompt = [
        SystemMessage(content="""Sos un asistente que responde preguntas 
        usando información real de internet. 
        Usá la información de búsqueda para dar una respuesta clara y útil.
        Respondé en español."""),
        HumanMessage(content=f"""Pregunta: {estado['pregunta']}
        
Información encontrada en internet:
{estado['busqueda']}

Respondé la pregunta usando esta información.""")
    ]
    resultado = llm.invoke(prompt)
    return {"respuesta": resultado.content}

# 5. GRAFO
grafo = StateGraph(Estado)
grafo.add_node("buscar", buscar_en_web)
grafo.add_node("responder", generar_respuesta)
grafo.set_entry_point("buscar")
grafo.add_edge("buscar", "responder")
grafo.add_edge("responder", END)
agente = grafo.compile()

# 6. EJECUTAR
if __name__ == "__main__":
    print("=== Agente con Búsqueda Web ===")
    print("Preguntame cualquier cosa actual\n")
    
    while True:
        pregunta = input("Tu pregunta (o 'salir'): ").strip()
        if pregunta.lower() == "salir":
            break
        
        resultado = agente.invoke({
            "pregunta": pregunta,
            "busqueda": "",
            "respuesta": ""
        })
        
        print(f"\nRespuesta: {resultado['respuesta']}\n")
        print("-" * 40)