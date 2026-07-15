import time
from dataclasses import dataclass, field
from datetime import datetime

from rag_helper import RAGBase

@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)

def calculate_cost(model, usage):
    cost = 0
    haiku_pricing_input = 1
    haiku_pricing_output = 5
    
    if "claude-haiku-4-5" in model:
        cost = (
            usage.input_tokens * haiku_pricing_input
            + usage.output_tokens * haiku_pricing_output
        ) / 1_000_000
    
    return cost

class RAGWithMetrics(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_call: LLMCallRecord = None

    def llm(self, prompt):
        start_time = time.time()
        response = self._call_llm(prompt)
        response_time = time.time() - start_time
        self._log_response(prompt, response, response_time)
        return response
    
    def _call_llm(self, prompt):
        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.instructions,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return response
    
    def _log_response(self, prompt, response, response_time):
        usage = response.usage
        cost = calculate_cost(self.model, usage)
        answer = response.content[0].text

        call_record = LLMCallRecord(
            model=self.model,
            prompt=prompt,
            instructions=self.instructions,
            answer=answer,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=(usage.input_tokens + usage.output_tokens),
            response_time=response_time,
            cost=cost,
        )
    
        print(call_record)
        self.last_call = call_record