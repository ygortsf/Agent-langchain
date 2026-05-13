from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain
from langchain_core.messages import SystemMessage, HumanMessage
import re

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=300
)

math_chain = LLMMathChain.from_llm(llm=llm)

def eh_conta(texto):
    texto = texto.strip()
    return bool(re.fullmatch(r"[0-9+\-*/(). ]+", texto))

print("Agente iniciado. Digite 'sair' para encerrar.")

while True:
    pergunta = input("\nDigite algo: ").strip()

    if pergunta.lower() == "sair":
        break

    try:
        if eh_conta(pergunta):
            resposta = math_chain.invoke({"question": pergunta})
            print("\nResposta:")
            print(resposta["answer"])
        else:
            mensagens = [
                SystemMessage(content="""
Você é um assistente virtual altamente inteligente, analítico e preciso.

Regras obrigatórias:
- Responda sempre em português do Brasil.
- Nunca responda em inglês ou espanhol.
- Forneça apenas informações verdadeiras e coerentes.
- Não invente fatos, datas, nomes, referências ou eventos.
- Se não souber com confiança, diga claramente: "Não tenho certeza sobre essa informação."
- Prefira respostas objetivas, corretas e bem explicadas.
- Ao explicar conceitos, use linguagem clara e organizada.
- Ao responder perguntas históricas, científicas ou acadêmicas, priorize exatidão.
- Nunca crie respostas fictícias para completar lacunas.
- Se a pergunta estiver ambígua, peça esclarecimento.
- Seja útil sem enrolação.
"""),
                HumanMessage(content=pergunta)
            ]

            resposta = llm.invoke(mensagens)

            print("\nResposta:")
            print(resposta.content)

    except Exception as e:
        print("\nErro:")
        print(e)