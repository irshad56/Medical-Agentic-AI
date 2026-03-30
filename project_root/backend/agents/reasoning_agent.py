import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Initialize LLM with explicit API Key
llm = ChatGroq(
    temperature=0.4,  # Slightly higher temperature for more natural conversation
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)


def reasoning_agent(fused_data: dict, history: str):
    query = fused_data.get("query", "")
    analysis = fused_data.get("image_analysis")
    viz_result = fused_data.get("visualization")

    # Build Data Context only if analysis exists
    data_context = ""
    if analysis:
        results_str = "\n".join([f"- {res['label']}: {res['confidence']:.2%}" for res in analysis["top_3"]])
        data_context += f"VISION SYSTEM ANALYSIS:\n{results_str}\n"

    if viz_result:
        data_context += "SYSTEM NOTICE: A visualization chart was generated for this turn.\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a versatile Medical AI Assistant. Your goal is to be helpful, natural, and accurate.

        CRITICAL LOGIC:
        1. CONVERSATIONAL MODE: If the user is just saying hi, asking how you are, or making small talk, respond naturally and warmly. DO NOT force medical data or disclaimers into a simple "Hello."
        2. ANALYTICAL MODE: If the user asks about a skin condition, an image, or requests data, provide the clinical analysis based ONLY on the 'CONTEXT' provided. 
        3. DISCLAIMER RULE: Only include the 'Clinical Correlation Disclaimer' if you are actually providing a medical classification or analysis. Do not include it for "Hi" or "Who are you?".
        4. CLARITY: If there is an image uploaded but the user hasn't asked to classify it yet, don't analyze it until prompted.
        """),
        ("human", "CONTEXT DATA:\n{data_context}\n\nCONVERSATION HISTORY:\n{history}\n\nUSER QUERY: {query}")
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({
            "data_context": data_context if data_context else "No medical data provided for this turn.",
            "history": history,
            "query": query
        })
        return response.content
    except Exception as e:
        return f"I encountered an error processing your request: {str(e)}"