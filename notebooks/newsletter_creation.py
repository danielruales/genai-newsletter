# notebooks/newsletter_creation.py

import sys
import os
import argparse

from modules.env_vars import set_os_env_vars, check_missing_vars
from modules.neon_db import run_neon_query, load_sql_query
from modules.date_functions import get_current_date
from modules.reference_extraction import create_content_from_df
from modules.prompt_templates import one_shot_example, system_message_example
from modules.langchain_config import set_langsmith_client, get_langsmith_tracer, get_llm_model, load_model_costs

def create_newsletter():
    # Set environment variables
    set_os_env_vars()

    # Load the costs
    MODEL_COSTS = load_model_costs()

    # Initialize the language model
    model_name = "claude-3-5-sonnet-20241022"  # You can modify this to accept as an argument
    streaming = True  # Streaming is when the LLM returns a token at a time

    # Initialize the language model
    llm = get_llm_model(model_name, streaming, MODEL_COSTS)

    # Load the SQL query
    query = load_sql_query("web_pages.sql")
    df = run_neon_query(query)

    # print("Number of rows:", len(df.index))

    # Print out the results (summary, titles, etc.)
    all_content, all_content_list, all_content_dict = create_content_from_df(df)

    # System prompt
    newsletter_prompt = system_message_example()

    # Create the chain with tracing
    tracer = get_langsmith_tracer()
    chain = (newsletter_prompt | llm).with_config(
        {
            "callbacks": [tracer],
            "tags": ["newsletter_generation"],
        }
    )

    # Test the chain
    newsletter = chain.invoke({"context": all_content, "current_date": get_current_date()})
    # print(newsletter.content)
    return newsletter.content

def main():
    parser = argparse.ArgumentParser(description="Generate a newsletter from web pages.")
    parser.add_argument('--model', type=str, default="claude-3-5-sonnet-20241022", help='Name of the language model to use.')
    args = parser.parse_args()

    # You can modify the model_name based on the argument
    global model_name
    model_name = args.model

    newsletter_content = create_newsletter()
    print(newsletter_content)
    with open("newsletter_content.md", "w") as file:
        file.write(newsletter_content)

# python notebooks/newsletter_creation.py --model "claude-3-5-sonnet-20241022"
if __name__ == "__main__":
    main()