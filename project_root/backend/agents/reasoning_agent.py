import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    temperature=0.2,
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)


def reasoning_agent(fused_data: dict, history: str):
    query = fused_data.get("query", "")
    analysis = fused_data.get("image_analysis")
    viz_result = fused_data.get("visualization")

    data_context = ""
    if analysis:
        results_str = "\n".join([f"- {res['label']}: {res['confidence']:.2%}" for res in analysis["top_3"]])
        data_context += f"VISION SYSTEM ANALYSIS:\n{results_str}\n"

    if viz_result:
        data_context += "SYSTEM NOTICE: A visualization chart was generated.\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a Dermatological AI Specialist. 
        Provide a concise, professional response based ONLY on the provided VISION ANALYSIS and the User Query.

        - If a chart was generated, mention it briefly.
        - Explain the top finding simply.
        - Always include the clinical correlation disclaimer at the end.
        - DO NOT add introductory filler text or greetings unless the user just said hi.
        """),
        ("human", "CONTEXT:\n{data_context}\n\nHISTORY:\n{history}\n\nQUERY: {query}")
    ])

    chain = prompt | llm
    response = chain.invoke({"data_context": data_context, "history": history, "query": query})
    return response.content