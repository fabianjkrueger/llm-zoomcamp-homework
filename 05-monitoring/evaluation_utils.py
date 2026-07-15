import time

from tqdm.auto import tqdm
from rag_helper import RAGBase


def calc_price(usage):
    input_price_per_million = 1.00
    output_price_per_million = 5.00

    input_cost = (usage.input_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def calc_total_price(usages):
    total_cost = 0.0

    for usage in usages:
        cost = calc_price(usage)
        total_cost = total_cost + cost["total_cost"]

    return total_cost


def llm_structured(
    client,
    instructions,
    user_prompt,
    output_type,
    model="claude-haiku-4-5",
    max_tokens=1024,
):
    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=instructions,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        output_format=output_type,
    )

    return response.parsed_output, response.usage


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="claude-haiku-4-5",
    max_retries=3,
    max_tokens=1024,
):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
                max_tokens=max_tokens,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
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

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.content[0].text

    def total_cost(self):
        return calc_total_price(self.usages)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results
