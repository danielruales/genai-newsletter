from dotenv import load_dotenv
import os

def set_os_env_vars():
    # Access keys and configurations
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv('LANGCHAIN_ENDPOINT')
    os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
    os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGCHAIN_PROJECT')
    os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
    os.environ["NOTION_API_KEY"]=os.getenv('NOTION_API_KEY')
    os.environ["LIBRARY_DATABASE_ID"]=os.getenv('LIBRARY_DATABASE_ID')
    os.environ["NEON_DATABASE_URL"] = os.getenv("NEON_DATABASE_URL")


required_vars = [
    'LANGCHAIN_API_KEY',
    'ANTHROPIC_API_KEY',
    'LIBRARY_DATABASE_ID',
    'NEON_DATABASE_URL'
]

def check_missing_vars(required_vars=required_vars):
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    error_msg = None
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        # raise EnvironmentError(error_msg)
    return error_msg
