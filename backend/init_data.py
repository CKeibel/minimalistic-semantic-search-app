from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    Hits
)
import pandas as pd
from typing import List, Tuple
from torch import FloatTensor
from embedding_pipeline import SentenceEmbeddingPipeline
from transformers import AutoTokenizer, AutoModel


_VECTOR_DIM = 384
model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipeline = SentenceEmbeddingPipeline(model=model, tokenizer=tokenizer)

data_path = "/Users/christopherkeibel/Documents/Workspace/own_projects/semantic_search/backend/db/goodreads_data.csv"


def load_data() -> pd.DataFrame:
    return pd.read_csv(data_path).iloc[:100]


def process_data(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[float], List[int], List[str], List[FloatTensor]]:
    titles = []
    authors = []
    descriptions = []
    ratings = []
    num_ratings = []
    urls = []
    embeddings = []

    

    for idx, row in df.iterrows():
        titles = titles + [row["Book"]]
        authors = authors + [row["Author"]]
        descriptions = descriptions + [row["Description"]]
        ratings = ratings + [row["Avg_Rating"]]
        num_ratings = num_ratings + [row["Num_Ratings"]]
        urls = urls + [row["URL"]]
        embeddings = embeddings + [pipeline(row["Description"])[0].numpy()]

    return (titles, authors, descriptions, ratings, num_ratings, urls, embeddings)


def main():
    connections.connect("default", host="localhost", port="19530")
    utility.drop_collection("goodreads")
    
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=10_000),
        FieldSchema(name="rating", dtype=DataType.DOUBLE),
        FieldSchema(name="num_ratings", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=_VECTOR_DIM)
    ]
    schema = CollectionSchema(fields)
    db = Collection("goodreads", schema)


    df = load_data()
    titles, authors, descriptions, ratings, num_ratings, urls, embeddings = process_data(df)

 
    insert_result = db.insert([
        titles, authors, descriptions, ratings, num_ratings, urls, embeddings
    ])
    db.flush()  


    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    db.create_index("embeddings", index)
    """
    db.load()
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 3},
    }
    search = pipeline("A Boy who became a wizard")[0].numpy()
    results = db.search([search], "embeddings", search_params, limit=3, output_fields=["title"])
    hits = Hits(results[0])

    for re in iter(hits):
        print(re.entity.title)
    """


if __name__ == "__main__":
    main()