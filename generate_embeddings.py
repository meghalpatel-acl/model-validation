import os
import json
import csv
from llama_cpp import Llama


TEST_DIR = os.path.dirname(__file__)
DATA = os.path.join(TEST_DIR, "dishwasher.json")
OUTPUT_CSV = os.path.join(TEST_DIR, "question_embeddings.csv")
MODEL_PATH = os.path.join(TEST_DIR, "models/all-MiniLM-L6-v2-Q8_0.gguf")


# Initialize embedding model
llm = Llama(
    model_path=MODEL_PATH,
    n_threads=4,
    n_ctx=128,
    n_batch=512,
    embedding=True,
    verbose=False,
)


def load_questions(json_path: str):
    """
    Load JSON file and extract only questions.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_pairs = data.get("qa_pairs", [])
    questions = [item["question"] for item in qa_pairs if "question" in item]

    return questions


def generate_embedding(text: str):
    """
    Generate embedding for a given text.
    """
    try:
        embedding = llm.embed(text)

        if not isinstance(embedding, list):
            raise ValueError("Embedding is not a list.")

        return embedding

    except Exception as e:
        print(f"Error generating embedding for text: {text}")
        print(f"Reason: {e}")
        return None


def save_to_csv(question_embedding_pairs, output_path):
    """
    Save (question, embedding) pairs to CSV.
    """
    if not question_embedding_pairs:
        print("No embeddings to save.")
        return

    embedding_size = len(question_embedding_pairs[0][1])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["question"] + [f"dim_{i}" for i in range(embedding_size)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for question, embedding in question_embedding_pairs:
            row = {"question": question}
            row.update({f"dim_{i}": val for i, val in enumerate(embedding)})
            writer.writerow(row)

    print(f"Saved embeddings to: {output_path}")


def main():
    # 1️⃣ Load questions
    questions = load_questions(DATA)
    print(f"Loaded {len(questions)} questions.")

    # 2️⃣ Generate embeddings
    question_embedding_pairs = []

    for idx, question in enumerate(questions):
        print(f"Processing {idx + 1}/{len(questions)}")

        embedding = generate_embedding(question)

        if embedding:
            question_embedding_pairs.append((question, embedding))

    print(f"Generated {len(question_embedding_pairs)} embeddings.")

    # 3️⃣ Save to CSV
    save_to_csv(question_embedding_pairs, OUTPUT_CSV)


if __name__ == "__main__":
    main()
