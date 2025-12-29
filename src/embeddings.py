# src/embeddings.py
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any
import torch

class EmbeddingManagerHF:
    def __init__(self, model_name='BAAI/bge-base-en-v1.5'):
        """
        Инициализация менеджера эмбеддингов
        
        Варианты моделей:
        - 'sentence-transformers/all-MiniLM-L6-v2' (быстрая, 384-dim)
        - 'BAAI/bge-base-en-v1.5' (хорошее качество, 768-dim)
        - 'intfloat/e5-base-v2' (универсальная)
        - 'allenai/specter' (для научных текстов)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model on {self.device}...")
        
        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )
        
        # Настройка ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./data/embeddings/chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Создание коллекции
        self.collection = self.chroma_client.get_or_create_collection(
            name="alzheimer_research",
            metadata={
                "hnsw:space": "cosine",
                "description": "Alzheimer's disease research articles"
            }
        )
        
        self.batch_size = 32  # Размер батча для кодирования
    
    def create_embeddings_batch(self, texts: List[str], metadatas: List[Dict]):
        """Создание эмбеддингов батчами"""
        all_embeddings = []
        all_metadatas = []
        all_documents = []
        all_ids = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_metadatas = metadatas[i:i+self.batch_size]
            
            # Создание эмбеддингов
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Нормализация для косинусного сходства
            )
            
            # Подготовка данных для добавления
            for j, (text, metadata) in enumerate(zip(batch_texts, batch_metadatas)):
                idx = i + j
                all_embeddings.append(batch_embeddings[j].tolist())
                all_documents.append(text)
                all_metadatas.append(metadata)
                all_ids.append(f"chunk_{idx}")
            
            print(f"Processed {i+len(batch_texts)}/{len(texts)} chunks")
        
        # Добавление в ChromaDB
        self.collection.add(
            embeddings=all_embeddings,
            documents=all_documents,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        print(f"Added {len(all_documents)} chunks to vector database")
        return all_embeddings
    
    def search_similar(self, query: str, n_results: int = 5, 
                      filter_conditions: Dict = None) -> Dict:
        """Поиск похожих документов"""
        # Кодирование запроса
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Поиск в векторной базе
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filter_conditions,
            include=["documents", "metadatas", "distances", "embeddings"]
        )
        
        # Конвертация расстояний в схожести
        for i, distance in enumerate(results['distances'][0]):
            results['distances'][0][i] = 1 - distance
        
        return results
    
    def hybrid_search(self, query: str, n_results: int = 5):
        """Гибридный поиск (семантический + ключевые слова)"""
        # Семантический поиск
        semantic_results = self.search_similar(query, n_results=n_results)
        
        # Поиск по ключевым словам (опционально)
        # Можно добавить BM25 или другие методы
        
        return semantic_results