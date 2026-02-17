import os
import pandas as pd
import numpy as np
import json
import csv
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity
import hashlib


TEST_DIR = os.path.dirname(__file__)
DATA = f"{TEST_DIR}/data"
REVIEW_CVS = f"{TEST_DIR}/review/similarity_matter_all.csv"
MODEL_PATH = f"{TEST_DIR}/models/all-MiniLM-L6-v2-Q8_0.gguf"

llm = Llama(
            model_path=MODEL_PATH,
            n_threads=4,
            n_ctx=128,
            n_batch=512,
            embedding=True,
            verbose=False,  
        )

def file_checksum(content: str, hash_length: int = 16) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:hash_length]

def generate(text):
        """
        Generates embeddings for the given text using the quantized model.

        Args:
            text (str): The input text for which to generate embeddings.

        Returns:
            list: A list representing the generated embedding.
        """
        try:
            embedding = llm.embed(text)
            if embedding is None:
                raise ValueError("No embedding returned")
            return embedding

        except Exception as e:
            print(f"Error generating embedding: {e}")
            exit(1)

def load_data(data_dir:str):
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    json_data = []
    for filename in json_files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath,"r") as file:
            data = json.load(file)
            qa_data = data.get("qa_pairs",[])

        json_data.extend(qa_data)
    return json_data

def load_embeddings(qa_pairs):
        filename = f"{TEST_DIR}/cached/vectors-{file_checksum(json.dumps(qa_pairs, sort_keys=True))}.tsv"

        if os.path.exists(filename):
            embeddings = pd.read_csv(filename, sep="\t", header=None).values
            return embeddings

        print("Generating question embeddings...")
        questions = [pair["question"] + " " + pair["answer"] for pair in qa_pairs]
        question_embeddings = np.array(
            [generate(q) for q in questions]
        )

        embedding_df = pd.DataFrame(question_embeddings)
        embedding_df.to_csv(filename, sep="\t", index=False, header=False)

        return question_embeddings

def main():        
        pairs = []
        data = load_data(DATA)
        questions = load_embeddings(data)
        similarity_matrix = cosine_similarity(questions)

        for i in range(len(questions)):
             for j in range(i+1, len(questions)):
                  if data[i].get("question") != data[j].get("question"):
                         pairs.append({
                    "question_1": data[i].get("question"),
                    "question_2": data[j].get("question"),
                    "cosine_similarity": similarity_matrix[i, j]
                  })

        if not pairs:
            print("No similarity pairs were generated. The process is complete.")
            return 

        pairs.sort(key=lambda x: x['cosine_similarity'], reverse=True) 

        with open(REVIEW_CVS, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["question_1", "question_2", "cosine_similarity"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
        
            writer.writeheader()
            writer.writerows(pairs)

        print(f"file saved at {REVIEW_CVS}.")


if __name__=="__main__":
    main()
