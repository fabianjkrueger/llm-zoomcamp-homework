from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# load env before importing starter (starter creates the Anthropic client)
from dotenv import load_dotenv

load_dotenv()

provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(ConsoleSpanExporter())
)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")

# after OTel is registered — starter builds index / client / plain rag
from starter import RAGBase, index, client

# subclass RAGBase and wrap steps/operations in spans
class RAGTraced(RAGBase):
    # module-level tracer is enough; no need to store it on self
    # so, no new __init__
    
    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search"):
            return super().search(query, num_results=num_results)
    
    def llm(self, prompt):
        with tracer.start_as_current_span("llm"):
            return super().llm(prompt)
    
    def rag(self, query):
        with tracer.start_as_current_span("rag"):
            return super().rag(query)

# initiate a rag assistant from the subclass
assistant = RAGTraced(
    index=index,
    llm_client=client,
)

# define query
query = "How does the agentic loop keep calling the model until it stops?"

# run rag on the query with trace and spans
assistant.rag(query)

