# Importing the LLM providers
from langchain_openai import ChatOpenAI # OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI # Google
from langchain_anthropic import ChatAnthropic # Anthropic

# Importing the prompt template and chains
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Importing LangSmith for tracing
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer

import os
import json
from pathlib import Path

def load_model_costs(config_path="../config/model_costs.json"):
    try:
        with open(Path(config_path)) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model costs file not found at {config_path}")

# Initialize LangSmith client
def set_langsmith_client():
    langsmith_client = Client()

# Initialize LangSmith tracer
def get_langsmith_tracer():
    tracer = LangChainTracer(project_name=os.getenv('LANGCHAIN_PROJECT'))
    return tracer

# Initialize the language model
def get_llm_model(model_name="claude-3-5-sonnet-20241022", streaming=False, MODEL_COSTS=[]):
    llm = None
    if model_name not in MODEL_COSTS:
        raise ValueError(f"Model {model_name} not found in the model costs")
    elif MODEL_COSTS[model_name]["provider"] == 'anthropic':
        llm = ChatAnthropic(
            model=model_name,
            max_tokens=4096,
            temperature=0.3,
            streaming=streaming,
            tags=["newsletter"],
            metadata={
                "ls_provider": "anthropic",
                "ls_model_name": model_name,
                "model_name": model_name,
                "model_cost_per_1k_input_tokens": MODEL_COSTS[model_name]["input"],   # price per 1K input tokens
                "model_cost_per_1k_output_tokens": MODEL_COSTS[model_name]["output"]    # price per 1K output tokens
            }
        )
    elif MODEL_COSTS[model_name]["provider"] == 'openai':
        llm = ChatOpenAI(
            model=model_name,
            max_tokens=4096,
            temperature=0.3,
            streaming=streaming,
            tags=["newsletter"],
            metadata={
                "ls_provider": "anthropic",
                "ls_model_name": model_name,
                "model_name": model_name,
                # No explicit prices are needed for OpenAI models because the prices are built into LangSmith
                # "model_cost_per_1k_input_tokens": MODEL_COSTS[model_name]["input"],   # price per 1K input tokens
                # "model_cost_per_1k_output_tokens": MODEL_COSTS[model_name]["output"]    # price per 1K output tokens
            }
        )
    elif MODEL_COSTS[model_name]["provider"] == 'google':
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            max_tokens=4096,
            temperature=0.3,
            streaming=streaming,
            tags=["newsletter"],
            metadata={
                "ls_provider": "anthropic",
                "ls_model_name": model_name,
                "model_name": model_name,
                "model_cost_per_1k_input_tokens": MODEL_COSTS[model_name]["input"],   # price per 1K input tokens
                "model_cost_per_1k_output_tokens": MODEL_COSTS[model_name]["output"]    # price per 1K output tokens
            }
        )
    return llm