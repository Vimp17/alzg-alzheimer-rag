# src/data_collection.py
from pymed import PubMed
import pandas as pd
import json
from datetime import datetime
import time

class PubMedCollectorHF:
    def __init__(self, email="matveychudaakov@gmail.com"):
        self.pubmed = PubMed(tool="AlzheimerRAG", email=email)
        
    def search_articles(self, queries, max_results=100):
        """Поиск статей по нескольким запросам"""
        all_articles = []
        
        search_queries = [
            '("Alzheimer\'s disease" AND (target OR therapeutic target OR drug target))',
            '("tau protein" AND (inhibitor OR modulator OR therapeutic))',
            '("amyloid beta" AND (clearance OR degradation OR inhibitor))',
            '("neuroinflammation" AND Alzheimer AND (target OR pathway))',
            '("APOE4" AND (therapy OR treatment OR target))',
            '("mitochondrial dysfunction" AND Alzheimer AND target)',
            '("synaptic plasticity" AND Alzheimer AND therapeutic)'
        ]
        
        for query in search_queries[:5]:  # Первые 5 запросов
            print(f"Searching for: {query}")
            results = self.pubmed.query(query, max_results=20)
            
            for article in results:
                try:
                    article_data = {
                        'pubmed_id': article.pubmed_id,
                        'title': article.title,
                        'abstract': article.abstract if article.abstract else '',
                        'keywords': article.keywords if article.keywords else [],
                        'journal': article.journal,
                        'publication_date': str(article.publication_date),
                        'authors': article.authors,
                        'doi': article.doi if article.doi else '',
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{article.pubmed_id}/"
                    }
                    all_articles.append(article_data)
                except:
                    continue
            
            time.sleep(1)  # Уважаем rate limits
        
        return all_articles[:max_results]

# Пример использования
collector = PubMedCollectorHF()
articles = collector.search_articles(max_results=50)
df = pd.DataFrame(articles)
df.to_json('data/raw/alzheimer_articles.json', orient='records', indent=2)