import pymupdf
import json

with pymupdf.open("data/thesis_report.pdf") as doc:
    for i, page in enumerate(doc):
        text = page.get_text()
        print(f"Σελίδα {i+1}: {len(text)} χαρακτήρες")

with pymupdf.open("data/thesis_report.pdf") as doc:
    print(doc[85].get_text())  # index 85 = σελίδα 86


with open("data/chunks.json", "r") as f:
    chunks = json.load(f)

keyword = "DOC_FLARE"  # λέξη που υπάρχει στα αποτελέσματα
for i, chunk in enumerate(chunks):
    if keyword in chunk:
        print(f"Chunk {i}: {chunk[:200]}")