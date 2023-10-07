import os
import openai
import pandas as pd
from langchain import OpenAI
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusClient
)
import pickle
from tenacity import retry, wait_random_exponential, stop_after_attempt


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def milvus_data_preprocess(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def construct_database():
    data = milvus_data_preprocess('../tool_embedding.pkl')
    data = [{'tool': el['tool'], 'embedding': el['embedding']} for el in data if el['tool'] != 'legal_document_retrieval' and el['tool'] != 'LawyerPR_PreliminaryReview']
    connections.connect("default", host="localhost", port="19530")
    tool_name = FieldSchema(name='tool', dtype=DataType.VARCHAR, is_primary=True, max_length=128)
    embedding = FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, is_primary=False, dim=1536)
    schema = CollectionSchema(fields=[tool_name, embedding], description='tool embedding')
    collection_name = 'tool_embedding'
    collection = Collection(name=collection_name, schema=schema, using='default')
    tool_name = [el['tool'] for el in data]
    embedding = [el['embedding'] for el in data]
    mr = collection.insert([tool_name, embedding])
    index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    print(mr)


def search(embedding, limit_num=50):
    collection = Collection(name='tool_embedding', using='default')
    print('Loading Milvus Database...')
    collection.load()
    search_params = {"metric_type": "L2", "params": {"nprobe": 20}}
    res = collection.search(data=embedding, param=search_params, anns_field="embedding",
                            limit=limit_num, expr=None, output_fields=['tool'])
    return res[0]


def get_excluded_list(string):
    connections.connect("default", host="localhost", port="19530")
    client = MilvusClient(url='http://localhost:19530')
    embedding = get_embedding(string)
    results = search([embedding], limit_num=30)
    excluded_list = [el.to_dict()['id'] for el in results]
    print(excluded_list)
    return excluded_list


def get_excluded_tool_list(tool):
    connections.connect("default", host="localhost", port="19530")
    client = MilvusClient(url='http://localhost:19530')
    embedding = client.get(collection_name='tool_embedding', ids=[tool])[0]['embedding']
    results = search([embedding], limit_num=20)
    excluded_list = [el.to_dict()['id'] for el in results]
    print(excluded_list)
    return excluded_list


if __name__ == '__main__':
    connections.connect("default", host="localhost", port="19530")
    utility.drop_collection("tool_embedding")
    construct_database()
