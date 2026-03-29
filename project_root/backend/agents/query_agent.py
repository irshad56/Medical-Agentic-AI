import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()

# Initialize LLM with explicit API Key
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)


def query_understanding_agent(query: str):
    """
    Uses LLM to dynamically determine if the user wants
    analysis, visualization, or just a chat.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are the Intent Router for a Medical Agentic OS. 
        Analyze the user query and return ONLY a JSON object.

        RULES:
        1. If they ask to identify, classify, or look at the image: "requires_ml": true
        2. If they ask for a chart, graph, plot, or visualization: "requires_viz": true
        3. If they are just saying hi or asking a general question: set both to false.

        JSON FORMAT:
        {{
            "intent": "string description",
            "requires_ml": boolean,
            "requires_viz": boolean
        }}
        """),
        ("human", "{query}")
    ])

    chain = prompt | llm
    try:
        response = chain.invoke({"query": query})
        # Clean potential markdown backticks
        json_str = response.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(json_str)
    except Exception as e:
        print(f"Query Agent Error: {e}")
        return {"intent": "fallback", "requires_ml": True, "requires_viz": False}