import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os


load_dotenv()
index = faiss.read_index("data/faiss_index.bin")
model = SentenceTransformer("intfloat/multilingual-e5-large")

with open("data/chunks.json","r") as file:
    chunks = json.load(file)

print(f"Index loaded: {index.ntotal} vectors")
print(f"Chunks loaded: {len(chunks)}")


# question = "Ποια μέθοδος χρησιμοποιήθηκε για την ανάλυση του HRV;"

hf_token = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=hf_token)


def answer_question(question):

    translation_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=[{"role": "user", "content": f"Translate the following question to Greek. Return only the translation, nothing else: {question}"}],
    max_tokens=200
)
    question = translation_response.choices[0].message.content.strip()
    print(f"Μεταφρασμένη ερώτηση: {question}")

    question_embedding = model.encode(["query: " + question])
    question_embedding = np.array(question_embedding).astype("float32")

    distances, indices = index.search(question_embedding, k=3)

    print("Top 3 chunks:")
    for i, idx in enumerate(indices[0]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunks[idx])


    context = "\n\n".join([chunks[idx] for idx in indices[0]])
    print("\n--- Context που θα σταλεί στο LLM ---")
    print(context)


    

    prompt = f"""Απάντησε στην ερώτηση βασιζόμενος μόνο στο παρακάτω κείμενο.

    Κείμενο:
    {context}

    Ερώτηση: {question}

    Απάντηση:"""


    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )

    greek_answer = response.choices[0].message.content

    translation_response2 = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[{"role": "user", "content": f"Translate the following text to English. Return only the translation, nothing else:\n\n{greek_answer}"}],
        max_tokens=1000
    )

    english_answer = translation_response2.choices[0].message.content.strip()

    final_answer = f"🇬🇷 Ελληνικά:\n{greek_answer}\n\n🇬🇧 English:\n{english_answer}"

    print("\n--- Απάντηση LLM ---")
    print(final_answer)

    return final_answer
