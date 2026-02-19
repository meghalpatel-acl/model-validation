import os
import sys
import pandas as pd
import numpy as np
import json
import csv
from llama_cpp import Llama
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# DATA = "./dishwasher.json"
DATA = "./test_queries.json"
REVIEW_CVS = "./data/question_embeddings.csv"
MODEL_PATH = "./models/all-MiniLM-L6-v2-Q8_0.gguf"
STATUS_FILE = "./status.txt"
SIMILARITY_THRESHOLD = 0.80


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

            # if not isinstance(embedding, list) or len(embedding) != 512:
            if not isinstance(embedding, list):
                raise ValueError(f"Invalid embedding format or size for text: {text}. Expected 512 values, got {len(embedding) if isinstance(embedding, list) else 'unknown'}.")

            return embedding

        except Exception as e:
            print(f"Error generating embedding for text: {text}. Error: {e}")
            return None

def load_data(data_file: str):
    """
    Load questions from the JSON file.

    Args:
        data_file (str): Path to the JSON file.

    Returns:
        list: A list of questions extracted from the JSON file.
    """
    filepath = os.path.join("./", data_file)
    with open(filepath, "r") as file:
        data = json.load(file)
        qa_data = data.get("qa_pairs", [])

    questions = [pair["question"] for pair in qa_data if "question" in pair]
    return questions

def compare_embeddings(embedding1, embedding2):
    """
    Compare two embeddings using cosine similarity.

    Args:
        embedding1 (list): The first embedding.
        embedding2 (list): The second embedding.

    Returns:
        float: The cosine similarity score between the two embeddings.
    """

    # Reshape embeddings for cosine similarity calculation
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)

    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity

def save_status(status: str):
    """
    Save the current status to a text file.

    Args:
        status (str): The status message to be saved.
    """
    with open(STATUS_FILE, 'w') as f:
        f.write(status)

def main():
    if len(sys.argv) > 1:
        # Take question from command line
        questions = [" ".join(sys.argv[1:])]
        print("Using question from CLI input.")
    else:
        # Fallback to JSON file
        questions = load_data(DATA)
        print("Using questions from JSON file.")

    # Load questions from the JSON file
    print(f"Loaded {len(questions)} questions from the data file.")

    original_embeddings = {}

    if os.path.exists(REVIEW_CVS):
        print(f"Existing embeddings file found at {REVIEW_CVS}. Loading existing embeddings...")
        with open(REVIEW_CVS, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row["question"]
                embedding = [float(value) for key, value in row.items() if key.startswith("dim_")]
                original_embeddings[question] = embedding
        print(f"Loaded {len(original_embeddings)} existing embeddings.")
    else:
        print(f"No existing embeddings file found at {REVIEW_CVS}.")

    
    # Generate embeddings for each question
    status_changed = False
    matched_results = []
    print("Running inference and comparing with stored embeddings...")
    
    for i, question in enumerate(questions):
        embedding = generate(question)

        if embedding is None:
            continue

        if question in original_embeddings:
            stored_embedding = original_embeddings[question]
            similarity = compare_embeddings(embedding, stored_embedding)

            print(f"[MATCH FOUND] {question}")
            print(f"Similarity: {similarity:.4f}")

            if similarity >= SIMILARITY_THRESHOLD:
                matched_results.append((question, similarity))
                status_changed = True
        else:
            print(f"[NO MATCH] {question}")

    if matched_results:
        lines = [
            "EMBEDDING SIMILARITY MATCHES",
            f"Threshold: {SIMILARITY_THRESHOLD}",
            ""
        ]

        for q, sim in matched_results:
            lines.append(f"- {q} | Similarity: {sim:.4f}")

        new_status = "\n".join(lines)
    else:
        new_status = "NO_MATCH"

    old_status = ""
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            old_status = f.read()

    if new_status != old_status:
        save_status(new_status)
        print("status.txt updated.")
    else:
        print("No change in status.txt.")



if __name__=="__main__":
    main()
