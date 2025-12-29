# src/preprocessing.py
import re
from nltk.tokenize import sent_tokenize
import nltk
from bs4 import BeautifulSoup

nltk.download('punkt_tab')

class TextProcessor:
    def __init__(self):
        # Список стоп-слов для научных текстов
        self.stop_words = set([
            'however', 'therefore', 'moreover', 'furthermore',
            'conversely', 'nevertheless', 'nonetheless', 
            'accordingly', 'consequently', 'similarly'
        ])
        
        # Паттерны для извлечения секций
        self.section_patterns = {
            'abstract': r'abstract|summary',
            'introduction': r'introduction|background',
            'conclusion': r'conclusion|summary and conclusions|concluding remarks'
        }
    
    def clean_text(self, text):
        """Очистка научного текста"""
        if not isinstance(text, str):
            return ""
        
        # Удаление HTML/XML тегов
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        
        # Удаление URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Удаление email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Удаление цифр в скобках (ссылки на литературу)
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        
        # Замена последовательностей пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def split_into_chunks(self, text, chunk_size=400, overlap=50):
        """Разделение на перекрывающиеся чанки с учетом предложений"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Сохраняем overlap предложений для следующего чанка
                overlap_size = int(len(current_chunk) * overlap / chunk_size)
                current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else []
                current_length = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def extract_key_terms(self, text):
        """Извлечение ключевых терминов"""
        # Паттерны для научных терминов
        patterns = {
            'genes': r'\b[A-Z]{1,2}\d{1,3}[A-Z]?\d*\b|\b[A-Z]{3,10}\d?\b',
            'proteins': r'\b(p|P)[0-9]{5}\b|\b[A-Z][a-z]+(in|ase|oid)\b',
            'drugs': r'\b[A-Z][a-z]+(mab|nib|zumab|ximab|zomab)\b'
        }
        
        key_terms = {}
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                key_terms[category] = list(set(matches))
        
        return key_terms