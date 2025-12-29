from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

class RAGPipelineHF:
    def __init__(self, embedding_manager, response_generator):
        self.embedding_manager = embedding_manager
        self.response_generator = response_generator
        
    def query(self, question: str, n_sources: int = 5, 
              threshold: float = 0.5, include_summaries: bool = True) -> Dict[str, Any]:
        """
        Выполнение запроса через RAG pipeline с опцией включения кратких содержаний
        """
        
        print(f"Step 1: Retrieving relevant documents for: '{question}'")
        retrieval_results = self.embedding_manager.search_similar(
            question, 
            n_results=n_sources * 3  # Извлекаем больше для лучшего отбора
        )
        
        # Проверка на наличие результатов
        if not retrieval_results or not retrieval_results.get('documents'):
            print("No documents found in retrieval.")
            return {
                "question": question,
                "answer": "No relevant documents found in the database. Please make sure the database is populated with articles.",
                "sources": [],
                "confidence": 0.0,
                "context_used": [],
                "metadata": {
                    "n_sources_retrieved": 0,
                    "n_sources_used": 0,
                    "retrieval_method": "semantic_search",
                    "error": "No documents in database"
                }
            }
        
        # Проверка, что есть документы
        if not retrieval_results['documents'][0]:
            print("Empty documents list.")
            return {
                "question": question,
                "answer": "The search returned no results. The database might be empty.",
                "sources": [],
                "confidence": 0.0,
                "context_used": [],
                "metadata": {
                    "n_sources_retrieved": 0,
                    "n_sources_used": 0,
                    "retrieval_method": "semantic_search",
                    "error": "Empty documents list"
                }
            }
        
        print(f"Found {len(retrieval_results['documents'][0])} initial documents.")
        
        # 2. Re-ranking - улучшение релевантности
        print("Step 2: Re-ranking results...")
        try:
            reranked_results = self.rerank_with_cross_encoder(
                question, 
                retrieval_results
            )
        except Exception as e:
            print(f"Error in re-ranking: {e}")
            # Используем оригинальные результаты если реранжирование не удалось
            reranked_results = retrieval_results
        
        # 3. Filtering - фильтрация по порогу
        print("Step 3: Filtering results...")
        filtered_results = self.filter_by_threshold(
            reranked_results, 
            threshold
        )
        
        if not filtered_results['documents'][0]:
            return {
                "question": question,
                "answer": "No sufficiently relevant sources found. Please try rephrasing your question or lowering the similarity threshold.",
                "sources": [],
                "confidence": 0.0,
                "context_used": [],
                "metadata": {
                    "n_sources_retrieved": len(retrieval_results['documents'][0]),
                    "n_sources_used": 0,
                    "retrieval_method": "semantic_search",
                    "error": "Below threshold"
                }
            }
        
        print(f"After filtering: {len(filtered_results['documents'][0])} relevant documents.")
        
        # 4. Context aggregation - объединение контекста с улучшенной обработкой
        print("Step 4: Preparing context with summaries...")
        final_contexts, final_metadatas = self.aggregate_context_with_summaries(
            filtered_results, 
            max_chunks=n_sources,
            include_summaries=include_summaries
        )
        
        # 5. Generation - генерация ответа
        print("Step 5: Generating answer with source summaries...")
        generation_result = self.response_generator.generate_answer(
            question,
            final_contexts,
            final_metadatas
        )
        
        # 6. Source attribution - атрибуция источников с краткими содержаниями
        print("Step 6: Attributing sources with summaries...")
        attributed_sources = self.attribute_sources_with_summaries(
            generation_result['answer'],
            final_contexts,
            final_metadatas
        )
        
        # 7. Confidence calculation - расчет уверенности
        confidence = 0.7  # Базовая уверенность
        if filtered_results['distances'][0]:
            avg_similarity = np.mean(filtered_results['distances'][0])
            confidence = min(avg_similarity * 1.2, 0.95)  # Максимум 95%
        
        # 8. Создание итогового ответа с краткими содержаниями
        final_answer = self.enhance_answer_with_summaries(
            generation_result['answer'],
            attributed_sources,
            include_summaries
        )
        
        return {
            "question": question,
            "answer": final_answer,
            "sources": attributed_sources,
            "confidence": confidence,
            "context_used": final_contexts[:3],
            "metadata": {
                "n_sources_retrieved": len(retrieval_results['documents'][0]),
                "n_sources_used": len(final_contexts),
                "retrieval_method": "semantic_search + cross-encoder",
                "generation_model": self.response_generator.__class__.__name__,
                "include_summaries": include_summaries
            }
        }
    
    def rerank_with_cross_encoder(self, query, results):
        """Реранжирование с помощью кросс-энкодера"""
        try:
            from sentence_transformers import CrossEncoder
            
            # Проверка наличия документов
            if not results or not results.get('documents') or not results['documents'][0]:
                return results
            
            # Проверка, что есть хотя бы один документ
            if len(results['documents'][0]) == 0:
                return results
            
            # Загрузка модели для реранжирования
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            pairs = []
            for doc in results['documents'][0]:
                # Проверка что документ не None и не пустой
                if doc and len(doc.strip()) > 0:
                    pairs.append([query, doc])
            
            # Если нет пар для оценки
            if not pairs:
                return results
            
            # Получение скоринга
            scores = cross_encoder.predict(pairs)
            
            # Если scores пуст
            if len(scores) == 0:
                return results
            
            # Сортировка по убыванию счета
            sorted_indices = np.argsort(scores)[::-1]
            
            # Реранжирование результатов
            reranked = {
                'documents': [[results['documents'][0][i] for i in sorted_indices if i < len(results['documents'][0])]],
                'metadatas': [[results['metadatas'][0][i] for i in sorted_indices if i < len(results['metadatas'][0])]],
                'distances': [[results['distances'][0][i] for i in sorted_indices if i < len(results['distances'][0])]],
            }
            
            return reranked
            
        except Exception as e:
            print(f"Cross-encoder reranking failed: {e}")
            # Возвращаем оригинальные результаты в случае ошибки
            return results
    
    def filter_by_threshold(self, results, threshold=0.5):
        """Фильтрация результатов по порогу схожести"""
        filtered_docs = []
        filtered_metas = []
        filtered_dists = []
        
        # Проверка наличия данных
        if not results or not results.get('documents') or not results['documents'][0]:
            return {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
        
        for doc, meta, dist in zip(results['documents'][0], 
                                  results['metadatas'][0], 
                                  results['distances'][0]):
            if dist >= threshold:
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_dists.append(dist)
        
        return {
            'documents': [filtered_docs],
            'metadatas': [filtered_metas],
            'distances': [filtered_dists]
        }
    
    def aggregate_context_with_summaries(self, results, max_chunks=5, include_summaries=True):
        """Объединение контекста с созданием кратких содержаний"""
        unique_contexts = []
        unique_metadatas = []
        
        if not results or not results['documents'][0]:
            return [], []
        
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            # Проверка на близкое содержание
            is_duplicate = False
            for existing_doc in unique_contexts:
                # Простая проверка на дубликаты по первым 100 символам
                if doc[:100] in existing_doc or existing_doc[:100] in doc:
                    is_duplicate = True
                    break
            
            if not is_duplicate and len(unique_contexts) < max_chunks:
                # Добавляем или создаем краткое содержание
                if 'abstract' in meta:
                    # Используем abstract как summary
                    meta['summary'] = meta['abstract'][:300] + "..." if len(meta['abstract']) > 300 else meta['abstract']
                elif 'summary' not in meta:
                    # Создаем краткое содержание из первых предложений
                    sentences = doc.split('. ')
                    summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else doc[:200] + "..."
                    meta['summary'] = summary
                
                unique_contexts.append(doc)
                unique_metadatas.append(meta)
        
        return unique_contexts, unique_metadatas
    
    def attribute_sources_with_summaries(self, answer, contexts, metadatas):
        """Создание структурированной информации об источниках с краткими содержаниями"""
        sources = []
        
        if not contexts or not metadatas:
            return sources
        
        for i, (context, metadata) in enumerate(zip(contexts, metadatas)):
            # Получаем или создаем summary
            summary = metadata.get('summary', '')
            if not summary:
                sentences = context.split('. ')
                summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else context[:200] + "..."
            
            source_info = {
                "source_id": i + 1,
                "title": metadata.get('title', 'Unknown'),
                "authors": metadata.get('authors', 'Unknown authors'),
                "journal": metadata.get('journal', 'Unknown journal'),
                "year": metadata.get('year', 'Unknown year'),
                "doi": metadata.get('doi', ''),
                "url": metadata.get('url', ''),
                "summary": summary,  # Добавляем краткое содержание
                "excerpt": context[:300] + "..." if len(context) > 300 else context,
                "relevance_score": metadata.get('relevance_score', 0.9 - (i * 0.1)),
                "similarity": metadata.get('distance', 0.8 - (i * 0.1))
            }
            
            # Проверка, цитируется ли этот источник в ответе
            if f"[Source {i+1}]" in answer or f"[{i+1}]" in answer:
                source_info["cited"] = True
                source_info["citation_count"] = answer.count(f"[Source {i+1}]")
            else:
                source_info["cited"] = False
                source_info["citation_count"] = 0
            
            sources.append(source_info)
        
        return sources
    
    def enhance_answer_with_summaries(self, answer, sources, include_summaries=True):
        """Улучшение ответа добавлением кратких содержаний источников"""
        
        if not include_summaries or not sources:
            return answer
        
        # Добавляем раздел с источниками
        enhanced_answer = f"{answer}\n\n{'='*60}\n📚 **REFERENCED ARTICLES**\n{'='*60}\n"
        
        for source in sources:
            if source.get('cited', False):
                enhanced_answer += f"\n**📖 [Source {source['source_id']}] {source['title']}**\n"
                enhanced_answer += f"👥 *Authors:* {source['authors']}\n"
                enhanced_answer += f"📅 *Year:* {source['year']}\n"
                enhanced_answer += f"📝 *Summary:* {source['summary']}\n"
                enhanced_answer += f"📊 *Relevance:* {source['relevance_score']:.2f}/1.0\n"
                enhanced_answer += f"🔗 *Citations in answer:* {source['citation_count']}\n"
        
        # Добавляем неиспользованные источники
        uncited_sources = [s for s in sources if not s.get('cited', False)]
        if uncited_sources:
            enhanced_answer += f"\n{'='*60}\n📖 **ADDITIONAL RELEVANT ARTICLES**\n{'='*60}\n"
            for source in uncited_sources[:3]:  # Ограничиваем 3 статьями
                enhanced_answer += f"\n• **{source['title']}** ({source['year']}) - {source['summary'][:150]}...\n"
        
        enhanced_answer += f"\n{'='*60}\n"
        enhanced_answer += "📌 *Note: This response is based on analysis of research articles. Consult original sources for complete information.*"
        
        return enhanced_answer
    
    def get_article_summary(self, article_id: str) -> Dict[str, Any]:
        """Получение краткого содержания конкретной статьи"""
        # Поиск статьи по ID
        results = self.embedding_manager.search_similar(
            article_id, 
            n_results=1
        )
        
        if not results or not results['documents'][0]:
            return {"error": "Article not found"}
        
        metadata = results['metadatas'][0][0]
        document = results['documents'][0][0]
        
        # Создаем расширенное summary
        title = metadata.get('title', 'Unknown')
        authors = metadata.get('authors', 'Unknown')
        year = metadata.get('year', 'Unknown')
        
        # Извлекаем ключевые предложения
        sentences = document.split('. ')
        key_sentences = []
        
        # Ищем предложения с ключевыми терминами
        keywords = ['find', 'show', 'demonstrate', 'conclude', 'suggest', 'indicate', 'reveal']
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                key_sentences.append(sentence.strip())
        
        # Берем первые 3 ключевых предложения или первые 3 предложения
        if len(key_sentences) >= 3:
            summary = '. '.join(key_sentences[:3]) + '.'
        else:
            summary = '. '.join(sentences[:3]) + '.'
        
        return {
            "article_id": article_id,
            "title": title,
            "authors": authors,
            "year": year,
            "summary": summary,
            "full_context": document[:500] + "..." if len(document) > 500 else document,
            "metadata": metadata
        }