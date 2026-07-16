import sqlite3
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult
)

# load env before importing starter (starter creates the Anthropic client)
from dotenv import load_dotenv

load_dotenv()

class SQLiteSpanExporter(SpanExporter):

    def __init__(self, db_path="traces.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self):
        return True

provider = TracerProvider()
provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter("traces.db"))
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
        with tracer.start_as_current_span("llm") as span:
            response = super().llm(prompt)

            usage = response.usage
            span.set_attribute("input_tokens", usage.input_tokens)
            span.set_attribute("output_tokens", usage.output_tokens)

            # same haiku pricing as in the module's metrics.py
            cost = (
                usage.input_tokens * 1 + usage.output_tokens * 5
            ) / 1_000_000
            span.set_attribute("cost", cost)

            return response

    
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

