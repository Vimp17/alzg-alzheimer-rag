# src/indexing.py
import json
import pandas as pd
from tqdm import tqdm

def create_chunked_dataset(articles_file, output_file="data/processed/chunks.json"):
    """Создание набора чанков из статей"""
    with open(articles_file, 'r') as f:
        articles = json.load(f)
    
    processor = TextProcessor()
    all_chunks = []
    chunk_metadata = []
    
    for article in tqdm(articles, desc="Processing articles"):
        # Объединяем все текстовые поля
        full_text = f"{article.get('title', '')}\n\n"
        full_text += f"{article.get('abstract', '')}\n\n"
        
        # Очистка текста
        cleaned_text = processor.clean_text(full_text)
        
        # Разделение на чанки
        chunks = processor.split_into_chunks(cleaned_text)
        
        # Извлечение ключевых терминов
        key_terms = processor.extract_key_terms(cleaned_text)
        
        # Создание метаданных для каждого чанка
        for i, chunk in enumerate(chunks):
            chunk_id = f"{article.get('pubmed_id', 'unknown')}_chunk_{i}"
            
            all_chunks.append(chunk)
            chunk_metadata.append({
                'chunk_id': chunk_id,
                'article_id': article.get('pubmed_id', 'unknown'),
                'title': article.get('title', ''),
                'authors': article.get('authors', []),
                'journal': article.get('journal', ''),
                'year': article.get('publication_date', '')[:4] if article.get('publication_date') else '',
                'chunk_index': i,
                'total_chunks': len(chunks),
                'key_terms': key_terms,
                'doi': article.get('doi', ''),
                'url': article.get('url', '')
            })
    
    # Сохранение чанков
    chunk_data = {
        'chunks': all_chunks,
        'metadata': chunk_metadata
    }
    
    with open(output_file, 'w') as f:
        json.dump(chunk_data, f, indent=2)
    
    return all_chunks, chunk_metadata