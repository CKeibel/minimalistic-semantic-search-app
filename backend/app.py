from flask import Flask, request
from embedding_pipeline import SentenceEmbeddingPipeline
from transformers import AutoTokenizer, AutoModel


# Flask
app = Flask(__name__)

# transormers model
model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

@app.route("/")
def init():
    pipeline = SentenceEmbeddingPipeline(model=model, tokenizer=tokenizer)
    return str(pipeline("Hallo Welt!"))


@app.route("/search")
def search():
    pass


if __name__ == "__main__":
    app.run()