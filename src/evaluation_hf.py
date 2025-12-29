import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import json

class EvaluationMetrics:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        nltk.download('punkt', quiet=True)
    
    def calculate_retrieval_metrics(self, 
                                   query: str, 
                                   retrieved_docs: List[str],
                                   relevant_docs: List[str] = None) -> Dict[str, float]:
        """Метрики для retrieval"""
        metrics = {}
        
        # 1. Precision@K
        k = min(5, len(retrieved_docs))
        if relevant_docs:
            # Для демо - считаем все документы релевантными
            metrics['precision@5'] = min(len(retrieved_docs[:k]) / k, 1.0)
        
        # 2. Mean Reciprocal Rank (MRR)
        mrr = 0
        if relevant_docs:
            for i, doc in enumerate(retrieved_docs, 1):
                if doc in relevant_docs:
                    mrr = 1 / i
                    break
        metrics['mrr'] = mrr
        
        # 3. Average similarity score
        if retrieved_docs:
            query_embedding = self.embedding_model.encode(query)
            doc_embeddings = self.embedding_model.encode(retrieved_docs)
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
            metrics['avg_similarity'] = float(np.mean(similarities))
            metrics['max_similarity'] = float(np.max(similarities))
        
        # 4. Diversity of retrieved documents
        if len(retrieved_docs) > 1:
            doc_embeddings = self.embedding_model.encode(retrieved_docs)
            pairwise_similarities = cosine_similarity(doc_embeddings)
            np.fill_diagonal(pairwise_similarities, 0)
            metrics['diversity'] = float(1 - np.mean(pairwise_similarities))
        else:
            metrics['diversity'] = 1.0
        
        return metrics
    
    def calculate_generation_metrics(self,
                                    generated_answer: str,
                                    retrieved_docs: List[str],
                                    query: str) -> Dict[str, float]:
        """Метрики для generation"""
        metrics = {}
        
        # 1. Answer length metrics
        words = generated_answer.split()
        metrics['answer_length'] = len(words)
        metrics['answer_length_chars'] = len(generated_answer)
        
        # 2. Lexical diversity
        if words:
            metrics['lexical_diversity'] = len(set(words)) / len(words)
        else:
            metrics['lexical_diversity'] = 0
        
        # 3. Source citation metrics
        source_citations = []
        for i in range(1, 11):  # Проверяем цитирование источников 1-10
            if f"[Source {i}]" in generated_answer:
                source_citations.append(i)
        
        metrics['num_citations'] = len(source_citations)
        metrics['citation_density'] = metrics['num_citations'] / max(len(words), 1)
        
        # 4. Factual consistency (простая версия)
        # Проверяем ключевые термины из запроса в ответе
        query_terms = set(query.lower().split())
        answer_terms = set(generated_answer.lower().split())
        metrics['query_coverage'] = len(query_terms.intersection(answer_terms)) / max(len(query_terms), 1)
        
        # 5. ROUGE scores против каждого документа
        rouge_scores = []
        for doc in retrieved_docs[:3]:  # Ограничиваем для скорости
            scores = self.rouge_scorer.score(doc[:1000], generated_answer[:1000])
            rouge_scores.append({
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            })
        
        if rouge_scores:
            metrics['avg_rouge1'] = np.mean([s['rouge1'] for s in rouge_scores])
            metrics['avg_rouge2'] = np.mean([s['rouge2'] for s in rouge_scores])
            metrics['avg_rougeL'] = np.mean([s['rougeL'] for s in rouge_scores])
        
        # 6. Semantic similarity с запросом
        query_embedding = self.embedding_model.encode(query)
        answer_embedding = self.embedding_model.encode(generated_answer)
        metrics['query_answer_similarity'] = float(cosine_similarity(
            [query_embedding], [answer_embedding]
        )[0][0])
        
        return metrics
    
    def calculate_system_metrics(self,
                                retrieval_time: float,
                                generation_time: float,
                                total_time: float,
                                retrieved_count: int,
                                used_count: int) -> Dict[str, float]:
        """Метрики производительности системы"""
        return {
            'retrieval_time_seconds': retrieval_time,
            'generation_time_seconds': generation_time,
            'total_time_seconds': total_time,
            'retrieval_speed_docs_per_second': retrieved_count / max(retrieval_time, 0.001),
            'generation_speed_tokens_per_second': 100 / max(generation_time, 0.001),  # Примерное значение
            'documents_retrieved': retrieved_count,
            'documents_used': used_count,
            'retrieval_efficiency': used_count / max(retrieved_count, 1)
        }

class ComprehensiveEvaluator:
    def __init__(self, rag_pipeline):
        self.pipeline = rag_pipeline
        self.metrics_calculator = EvaluationMetrics()
        
        # Тестовые вопросы с ожидаемыми темами
        self.test_queries = [
            {
                'query': "What are GSK3 and CDK5 inhibitors used for?",
                'expected_topics': ['tau', 'protein', 'aggregation', 'inhibitors'],
                'min_answer_length': 50
            },
            {
                'query': "How does amyloid-beta clearance work?",
                'expected_topics': ['glymphatic', 'BACE1', 'gamma-secretase'],
                'min_answer_length': 50
            },
            {
                'query': "What is neuroinflammation in Alzheimer's?",
                'expected_topics': ['microglial', 'TREM2', 'inflammatory'],
                'min_answer_length': 50
            }
        ]
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Запуск комплексной оценки"""
        all_results = []
        
        for test_case in self.test_queries:
            print(f"\nEvaluating query: {test_case['query']}")
            
            # Запуск pipeline
            result = self.pipeline.generate_response(test_case['query'])
            
            # Расчет метрик
            retrieval_metrics = self.metrics_calculator.calculate_retrieval_metrics(
                test_case['query'],
                [doc['excerpt'] for doc in result.get('sources', [])]
            )
            
            generation_metrics = self.metrics_calculator.calculate_generation_metrics(
                result['answer'],
                [doc['excerpt'] for doc in result.get('sources', [])],
                test_case['query']
            )
            
            system_metrics = self.metrics_calculator.calculate_system_metrics(
                result['retrieval_time'],
                result['generation_time'],
                result['total_time'],
                result['retrieval_metrics']['documents_retrieved'],
                result['retrieval_metrics']['documents_used']
            )
            
            # Проверка качества ответа
            quality_metrics = self._assess_answer_quality(
                result['answer'],
                test_case
            )
            
            case_result = {
                'query': test_case['query'],
                'answer_preview': result['answer'][:200] + '...',
                'retrieval_metrics': retrieval_metrics,
                'generation_metrics': generation_metrics,
                'system_metrics': system_metrics,
                'quality_metrics': quality_metrics,
                'sources_used': len([s for s in result['sources'] if s['cited']]),
                'total_sources': len(result['sources'])
            }
            
            all_results.append(case_result)
        
        # Агрегация результатов
        return self._aggregate_results(all_results)
    
    def _assess_answer_quality(self, answer: str, test_case: Dict) -> Dict[str, float]:
        """Оценка качества ответа"""
        answer_lower = answer.lower()
        
        # Проверка ожидаемых тем
        topics_present = []
        for topic in test_case['expected_topics']:
            if topic.lower() in answer_lower:
                topics_present.append(topic)
        
        topic_coverage = len(topics_present) / len(test_case['expected_topics'])
        
        # Проверка структуры ответа
        has_citations = '[Source' in answer
        has_multiple_sentences = len(answer.split('. ')) > 2
        has_reasonable_length = len(answer.split()) >= test_case['min_answer_length']
        
        return {
            'topic_coverage': topic_coverage,
            'has_citations': float(has_citations),
            'has_multiple_sentences': float(has_multiple_sentences),
            'has_reasonable_length': float(has_reasonable_length),
            'quality_score': (topic_coverage + 
                            float(has_citations) + 
                            float(has_multiple_sentences) + 
                            float(has_reasonable_length)) / 4
        }
    
    def _aggregate_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Агрегация результатов по всем тестам"""
        aggregated = {
            'average_metrics': {},
            'per_query_results': all_results,
            'summary': {}
        }
        
        # Средние значения по всем метрикам
        metric_categories = ['retrieval_metrics', 'generation_metrics', 
                           'system_metrics', 'quality_metrics']
        
        for category in metric_categories:
            category_metrics = {}
            metrics_list = [r[category] for r in all_results]
            
            # Собираем все ключи метрик
            all_keys = set()
            for metrics in metrics_list:
                all_keys.update(metrics.keys())
            
            # Вычисляем средние
            for key in all_keys:
                values = [m.get(key, 0) for m in metrics_list if key in m]
                if values:
                    category_metrics[f'avg_{key}'] = float(np.mean(values))
                    category_metrics[f'std_{key}'] = float(np.std(values))
            
            aggregated['average_metrics'][category] = category_metrics
        
        # Сводная статистика
        aggregated['summary'] = {
            'total_queries_evaluated': len(all_results),
            'avg_answer_length': np.mean([r['generation_metrics']['answer_length'] for r in all_results]),
            'avg_citations_per_answer': np.mean([r['generation_metrics']['num_citations'] for r in all_results]),
            'avg_quality_score': np.mean([r['quality_metrics']['quality_score'] for r in all_results]),
            'avg_retrieval_time': np.mean([r['system_metrics']['retrieval_time_seconds'] for r in all_results]),
            'avg_total_time': np.mean([r['system_metrics']['total_time_seconds'] for r in all_results])
        }
        
        return aggregated
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Генерация читаемого отчета"""
        report = []
        report.append("=" * 80)
        report.append("RAG PIPELINE EVALUATION REPORT")
        report.append("=" * 80)
        
        report.append(f"\nTotal Queries Evaluated: {results['summary']['total_queries_evaluated']}")
        
        # Retrieval метрики
        report.append("\n" + "-" * 40)
        report.append("RETRIEVAL METRICS")
        report.append("-" * 40)
        retrieval = results['average_metrics']['retrieval_metrics']
        report.append(f"Average Similarity Score: {retrieval.get('avg_avg_similarity', 0):.3f}")
        report.append(f"Max Similarity: {retrieval.get('avg_max_similarity', 0):.3f}")
        report.append(f"Diversity Score: {retrieval.get('avg_diversity', 0):.3f}")
        report.append(f"MRR: {retrieval.get('avg_mrr', 0):.3f}")
        
        # Generation метрики
        report.append("\n" + "-" * 40)
        report.append("GENERATION METRICS")
        report.append("-" * 40)
        generation = results['average_metrics']['generation_metrics']
        report.append(f"Average Answer Length: {generation.get('avg_answer_length', 0):.1f} words")
        report.append(f"Average Citations: {generation.get('avg_num_citations', 0):.1f}")
        report.append(f"Lexical Diversity: {generation.get('avg_lexical_diversity', 0):.3f}")
        report.append(f"Query-Answer Similarity: {generation.get('avg_query_answer_similarity', 0):.3f}")
        report.append(f"ROUGE-L Score: {generation.get('avg_avg_rougeL', 0):.3f}")
        
        # Качество
        report.append("\n" + "-" * 40)
        report.append("QUALITY METRICS")
        report.append("-" * 40)
        quality = results['average_metrics']['quality_metrics']
        report.append(f"Topic Coverage: {quality.get('avg_topic_coverage', 0):.3f}")
        report.append(f"Has Citations: {quality.get('avg_has_citations', 0):.3f}")
        report.append(f"Answer Structure Score: {quality.get('avg_quality_score', 0):.3f}")
        
        # Производительность
        report.append("\n" + "-" * 40)
        report.append("PERFORMANCE METRICS")
        report.append("-" * 40)
        system = results['average_metrics']['system_metrics']
        report.append(f"Average Total Time: {system.get('avg_total_time_seconds', 0):.2f}s")
        report.append(f"Retrieval Time: {system.get('avg_retrieval_time_seconds', 0):.2f}s")
        report.append(f"Generation Time: {system.get('avg_generation_time_seconds', 0):.2f}s")
        report.append(f"Retrieval Efficiency: {system.get('avg_retrieval_efficiency', 0):.3f}")
        
        # Рекомендации
        report.append("\n" + "-" * 40)
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        if quality.get('avg_topic_coverage', 0) < 0.7:
            report.append("⚠️  Improve topic coverage by better retrieval or prompt engineering")
        
        if generation.get('avg_num_citations', 0) < 2:
            report.append("⚠️  Add more source citations in answers")
        
        if system.get('avg_total_time_seconds', 0) > 5:
            report.append("⚠️  Optimize pipeline for faster response times")
        
        if generation.get('avg_answer_length', 0) < 100:
            report.append("⚠️  Generate more comprehensive answers")
        
        report.append("\n" + "=" * 80)
        
        return '\n'.join(report)

# Главный скрипт для запуска оценки
if __name__ == "__main__":
    from rag_pipeline_complete import RAGPipeline
    
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    print("Setting up Evaluator...")
    evaluator = ComprehensiveEvaluator(pipeline)
    
    print("Running Evaluation...")
    results = evaluator.run_evaluation()
    
    # Генерация отчета
    report = evaluator.generate_report(results)
    print(report)
    
    # Сохранение результатов
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to evaluation_results.json")
    
    # Визуализация ключевых метрик
    print("\nKey Performance Indicators:")
    print(f"• Answer Quality Score: {results['summary']['avg_quality_score']:.3f}/1.0")
    print(f"• Average Response Time: {results['summary']['avg_total_time']:.2f}s")
    print(f"• Citation Rate: {results['summary']['avg_citations_per_answer']:.1f}/source per answer")
    print(f"• Retrieval Relevance: {results['average_metrics']['retrieval_metrics'].get('avg_avg_similarity', 0):.3f}")