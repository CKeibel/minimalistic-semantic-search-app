from pymilvus import (
    connections,
    Collection,
    Hits,
    Hit,
)
from typing import List, Dict
from numpy import ndarray


class Database:
    def __init__(self):
        connections.connect("default", host="localhost", port="19530")
        
        self.collection = Collection("goodreads")
        self.collection.load()


    def map_result(self, hit: Hit) -> Dict:
        return {
                "title": hit.entity.title,
                "author": hit.entity.author,
                "description": hit.entity.description
            }


    def get_all(self) -> List[Dict]:
        res = self.collection.query(expr= "pk >= 0", output_fields=["title", "author", "description"])
        return res
        

    def search_by_vector(self, search_vector: ndarray, num_results: int = 10) -> List[Dict]:
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": num_results},
        }
        results = self.collection.search([search_vector], "embeddings", search_params, limit=10, output_fields=["title", "author", "description"])
        hits = Hits(results[0])

        return list(map(self.map_result, hits))

    