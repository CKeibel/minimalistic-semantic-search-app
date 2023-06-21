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

@app.route("/")
def init():
    pipeline = SentenceEmbeddingPipeline(model=model, tokenizer=tokenizer)
    vector = pipeline("A boy who became a wizard")[0].numpy()
    
    return milvus_db.search_by_vector(vector)


@app.route("/search")
def search():
    pass


if __name__ == "__main__":
    app.run()