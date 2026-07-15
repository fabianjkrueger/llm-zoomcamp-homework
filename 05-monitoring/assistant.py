"""
Create an assistant that can answer questions about the FAQ.
"""

# dependencies
import sys

from dotenv import load_dotenv
from anthropic import Anthropic

from ingest import load_faq_data, build_index
from metrics import RAGWithMetrics
from db_save import save_conversation

# function to create assistant
def create_assistant():
    load_dotenv()

    documents = load_faq_data()
    index = build_index(documents)

    return RAGWithMetrics(
        index=index,
        llm_client=Anthropic(),
    )

# if this is executed directly, create and query the agent
if __name__ == "__main__":
    assistant = create_assistant()

    # if a query is provided, use it, otherwise use a default query
    query = "How do I join the course?"
    if len(sys.argv) > 1:
        query = sys.argv[1]

    # query the agent
    answer = assistant.rag(query)
    print(answer)
    save_conversation(assistant.last_call, query, "llm-zoomcamp")