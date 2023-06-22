from flask import Flask, request
from embedding_pipeline import SentenceEmbeddingPipeline
from transformers import AutoTokenizer, AutoModel
from database import Database


# Flask
app = Flask(__name__)

# transformers model
model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# milvus
milvus_db = Database()


@app.route("/all")
def all():
    return milvus_db.get_all()


@app.route("/search")
def search():
    query = request.args.get("query")
    pipeline = SentenceEmbeddingPipeline(model=model, tokenizer=tokenizer)
    vector = pipeline(query)[0].numpy()
    return milvus_db.search_by_vector(vector)


if __name__ == "__main__":
    app.run()