import os
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

DIMENSION = 768     # taille des vecteurs du modele nomic-embed-text 
COLLECTION_NAME = "YuGiOh_Rules"
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL =  "qwen2.5:1.5b"
DB_PATH = "./qdrant_db"
YUGIOH_RULES_FILE = Path(__file__).resolve().parent / "data" / "regles_yugioh.txt"



def get_embedding(text: str) -> list[float]:
    res = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout = 30,
    )
    res.raise_for_status()
    return res.json()["embeddings"][0]

def ask_llm(prompt:str) -> str:
    res = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout = 120,
    )
    res.raise_for_status()
    return res.json()["response"]

def load_and_chunk_file(file_path: str) -> list[str]:
    """Découpe le fichier de règles par blocs logiques (titres / paragraphes)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_chunks = content.split("\n\n") # Découpe par double saut de ligne
    chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 20]
    return chunks

def setup_knowledge_base(client: QdrantClient):
    """Initialise et indexe les règles si la collection n'existe pas encore."""
    chunks = load_and_chunk_file(YUGIOH_RULES_FILE)
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in collections:
        point_count = client.count(collection_name=COLLECTION_NAME, exact=True).count
        if point_count > 0:
            print(f"La collection '{COLLECTION_NAME}' existe déjà. Aucune action nécessaire.")
            return

        print(f"La collection '{COLLECTION_NAME}' est vide. Recréation de l'index...")
        client.delete_collection(collection_name=COLLECTION_NAME)

    print(f"Création de la collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
    )

    points = []
    print(f"Génération des embeddings, {len(chunks)} chunks...")

    for idx, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        points.append(PointStruct(id=idx, vector=vector, payload={"text": chunk}))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexation terminée, {len(points)} chunks ajoutés à la collection '{COLLECTION_NAME}'.")
    

def query_rules(client: QdrantClient, question: str, top_k: int = 3) -> str:
    """Fonction pour interroger la base de connaissances et obtenir une réponse a partir des règles stockées"""
    query_vector = get_embedding(question)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points
    context = "\n---\n".join([hit.payload["text"] for hit in results])

    prompt = f"""Tu es un arbitre officiel et expert du jeu de cartes Yu Gi Oh. 
    Tu réponds en français et de manière précise, uniquement à partir du contexte fourni.
    Si le contexte ne suffit pas, dis-le.
    
    Contexte: {context}

    Question: {question}
    Réponse:"""

    return ask_llm(prompt)

if __name__ == "__main__":
    client = QdrantClient(path=DB_PATH)

    """
    setup_knowledge_base(client)

    # Exemples d'usages
    test_questions = [
        "Connais-tu le jeu de cartes Yu Gi Oh ?",
        "J'ai 37 cartes dans mon Main deck, est-ce que c'est autorisé pour jouer ?",
        "Est-ce que je peux invoquer un monstre de niveau 8 avec seulement 1 monstre de niveau 4 sur le terrain ?",
    ]

    for q in test_questions:
        print(f"\nQuestion: {q}")
        reponse = query_rules(client, q)
        print(f"Réponse: \n{reponse}")
    """

    try:
        setup_knowledge_base(client)
        print("\nPosez vos questions sur Yu-Gi-Oh ! Tapez 'quit' pour quitter.")
    
        while True:
            try:
                question = input("\nQuestion : ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
    
            if question.lower() in {"quit", "exit", "q"}:
                break
            if not question:
                continue
    
            try:
                reponse = query_rules(client, question)
                print(f"Réponse :\n{reponse}")
            except requests.RequestException as error:
                print(f"Erreur de communication avec Ollama : {error}")
    finally:
        client.close()
