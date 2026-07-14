import requests
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm
from minsearch import Index, VectorSearch
from sentence_transformers import SentenceTransformer


def load_faq_data():
    docs_url = 'https://datatalks.club/faq/json/courses.json'
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = 'https://datatalks.club/faq'

    for course in courses_raw:
        course_url = f'{url_prefix}{course["path"]}'
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        documents.extend(course_data)

    for doc in documents:
        doc["doc_id"] = doc.pop("id") #we do this so we can add the id key to sqlite so we don't reimport the same records

    return documents


def build_text_index(documents):
    index = Index(
        text_fields=['content'],
        keyword_fields=['filename']
    )
    index.fit(documents)
    return index

def build_vector_index(documents):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [doc["content"] for doc in documents]

    batch_size = 50
    vectors = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        vectors.extend(model.encode(batch))

    X = np.array(vectors)

    vindex = VectorSearch(
        keyword_fields=["filename"],
    )
    vindex.fit(X, documents)
    return vindex

