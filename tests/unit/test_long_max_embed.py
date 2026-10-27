from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Longueur maximale :", model.max_seq_length)

texte = "Colle ici un chunk de MediQuery"
tokens = model.tokenizer(
    texte,
    add_special_tokens=True,
    truncation=False
)["input_ids"]

print("Tokens du chunk :", len(tokens))
print("Tokens conservés :", min(len(tokens), model.max_seq_length))