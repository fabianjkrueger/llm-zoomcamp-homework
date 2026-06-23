INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()

# helper function to convert vector to string for pgvector search
def vec_to_str(vector):
    return "[" + ",".join(str(x) for x in vector) + "]"

class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course='llm-zoomcamp',
        model='claude-haiku-4-5',
        max_tokens=1024,
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model
        self.max_tokens = max_tokens

    def search(self, query, num_results=5):
        boost_dict = {'question': 3.0, 'section': 0.5}
        filter_dict = {'course': self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['section'])
            lines.append('Q: ' + doc['question'])
            lines.append('A: ' + doc['answer'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):

        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.instructions,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        # return usage along the response for the homework
        return response

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        # pass down usage for homework
        return response.content[0].text, response.usage.input_tokens

# subclass for homework
class RAGBaseHomework(RAGBase):
    
    def search(self, query, num_results=5):

        return self.index.search(
            query,
            num_results=num_results,
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['content'])
            lines.append('')

        return '\n'.join(lines).strip()
    
# subclass for vector search
class RAGVector(RAGBase):
    # it needs an embedder, so add it to init
    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder
    
    # modify search function for vector search: embedd query before search
    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}
        
        return self.index.search(
            query_vector=query_vector,
            num_results=num_results,
            filter_dict=filter_dict,
        )

# subclass for pgvector search
class RAGPgVector(RAGBase):
    # it needs an embedder and a connection to the database
    def __init__(self, embedder, conn, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    # modify search function for pgvector search: embedd query before search
    # then execute connection using SQL code
    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        query_str = vec_to_str(query_vector)

        rows = self.conn.execute(
            """
            SELECT course, section, question, answer
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (self.course, query_str, num_results)
        ).fetchall()
        
        # return results as a list of dictionaries
        return [
            {"course": r[0], "section": r[1], "question": r[2], "answer": r[3]}
            for r in rows
        ]