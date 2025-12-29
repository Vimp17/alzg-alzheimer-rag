import json
import chromadb
from sentence_transformers import SentenceTransformer
import os
from chromadb.config import Settings

def init_database():
    """Инициализация базы данных с демо-данными и сохранением abstract"""
    print("Initializing database with demo data...")
    
    # Создаем директории
    os.makedirs("./data/embeddings", exist_ok=True)
    os.makedirs("./data/raw", exist_ok=True)
    
    # Демо-статьи о болезни Альцгеймера
    demo_articles = [
        {
            "pmid": "demo_001",
            "title": "Tau protein aggregation in Alzheimer's disease",
            "abstract": "This review discusses tau protein aggregation mechanisms and potential therapeutic targets including GSK3β and CDK5 inhibitors. The study examines phosphorylation patterns and their role in neurofibrillary tangle formation.",
            "authors": ["Smith J", "Johnson R"],
            "year": "2023",
            "journal": "Nature Reviews Neurology",
            "doi": "10.1038/s41582-023-00789-7",
            "url": "https://example.com/tau-review"
        },
        {
            "pmid": "demo_002", 
            "title": "Amyloid-beta clearance pathways and therapeutic interventions",
            "abstract": "Research on amyloid-beta clearance through the glymphatic system and potential drug targets including BACE1 inhibitors and gamma-secretase modulators. The study highlights novel clearance mechanisms.",
            "authors": ["Brown K", "Davis M"],
            "year": "2022",
            "journal": "Science",
            "doi": "10.1126/science.abc1234",
            "url": "https://example.com/amyloid-review"
        },
        # ... остальные статьи
    ]
    
    # Сохраняем демо-статьи
    with open('./data/raw/demo_articles.json', 'w') as f:
        json.dump(demo_articles, f, indent=2)
    
    print("Demo articles saved.")
    
    # Инициализация ChromaDB
    client = chromadb.PersistentClient(
        path="./data/embeddings/chroma_db",
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Очищаем коллекцию если существует
    try:
        client.delete_collection("alzheimer_research")
        print("Old collection deleted.")
    except:
        print("No existing collection found.")
    
    # Создаем новую коллекцию
    collection = client.create_collection(
        name="alzheimer_research",
        metadata={
            "hnsw:space": "cosine",
            "description": "Alzheimer's disease research articles"
        }
    )
    
    # Загружаем модель для эмбеддингов
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Создаем чанки из статей
    chunks = []
    metadatas = []
    
    for article in demo_articles:
        # Создаем чанки из заголовка и абстракта
        text = f"Title: {article['title']}\n\nAbstract: {article['abstract']}"
        
        # Разбиваем на чанки по предложениям
        sentences = text.split('. ')
        chunk = ""
        
        for sentence in sentences:
            if len(chunk) + len(sentence) < 500:
                chunk += sentence + ". "
            else:
                if chunk.strip():
                    chunks.append(chunk.strip())
                    # Преобразуем список авторов в строку
                    authors_str = ", ".join(article['authors']) if isinstance(article['authors'], list) else str(article['authors'])
                    metadatas.append({
                        'title': article['title'],
                        'authors': authors_str,
                        'journal': article['journal'],
                        'year': article['year'],
                        'doi': article['doi'],
                        'url': article['url'],
                        'article_id': article['pmid'],
                        'type': 'research_article',
                        'abstract': article['abstract'],  # Сохраняем полный abstract
                        'summary': article['abstract'][:200] + "..." if len(article['abstract']) > 200 else article['abstract']
                    })
                chunk = sentence + ". "
        
        if chunk.strip():
            chunks.append(chunk.strip())
            authors_str = ", ".join(article['authors']) if isinstance(article['authors'], list) else str(article['authors'])
            metadatas.append({
                'title': article['title'],
                'authors': authors_str,
                'journal': article['journal'],
                'year': article['year'],
                'doi': article['doi'],
                'url': article['url'],
                'article_id': article['pmid'],
                'type': 'research_article',
                'abstract': article['abstract'],  # Сохраняем полный abstract
                'summary': article['abstract'][:200] + "..." if len(article['abstract']) > 200 else article['abstract']
            })
    
    print(f"Created {len(chunks)} chunks from {len(demo_articles)} articles.")
    
    # Проверяем, что есть чанки для добавления
    if len(chunks) == 0:
        print("Error: No chunks created. Exiting.")
        return False
    
    # Создаем эмбеддинги
    print("Creating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    
    # Проверяем размерность эмбеддингов
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Добавляем в базу данных
    print("Adding to ChromaDB...")
    
    # Подготавливаем данные для добавления
    embeddings_list = embeddings.tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Проверяем соответствие размеров
    if len(embeddings_list) != len(chunks) or len(chunks) != len(metadatas):
        print(f"Error: Mismatch in data sizes: embeddings={len(embeddings_list)}, chunks={len(chunks)}, metadatas={len(metadatas)}")
        return False
    
    # Добавляем данные
    collection.add(
        embeddings=embeddings_list,
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Database initialized with {len(chunks)} chunks.")
    print(f"✅ Collection 'alzheimer_research' created.")
    
    # Проверяем количество документов
    count = collection.count()
    print(f"✅ Total documents in collection: {count}")
    
    return True

if __name__ == "__main__":
    success = init_database()
    if success:
        print("\n🎉 Database initialization completed successfully!")
        print("📚 Articles include full abstracts for better summarization.")
    else:
        print("\n❌ Database initialization failed!")