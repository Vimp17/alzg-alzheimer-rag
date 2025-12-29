import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from typing import Dict, Any

class MetricsVisualizer:
    def __init__(self, results_file='evaluation_results.json'):
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]
    
    def plot_retrieval_metrics(self):
        """Визуализация метрик retrieval"""
        retrieval_data = self.results['average_metrics']['retrieval_metrics']
        
        metrics_to_plot = {
            'avg_avg_similarity': 'Average Similarity',
            'avg_max_similarity': 'Max Similarity',
            'avg_diversity': 'Diversity',
            'avg_mrr': 'Mean Reciprocal Rank'
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (metric_key, metric_name) in enumerate(metrics_to_plot.items()):
            if metric_key in retrieval_data:
                ax = axes[idx]
                value = retrieval_data[metric_key]
                std = retrieval_data.get(f'std_{metric_key[4:]}', 0)
                
                ax.bar(['Value'], [value], yerr=[std], 
                      capsize=10, color='skyblue', alpha=0.7)
                ax.set_title(f'{metric_name}: {value:.3f} ± {std:.3f}')
                ax.set_ylim(0, 1.1)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Retrieval Metrics Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('retrieval_metrics.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_generation_metrics(self):
        """Визуализация метрик generation"""
        generation_data = self.results['average_metrics']['generation_metrics']
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        metrics = [
            ('avg_answer_length', 'Answer Length (words)', 'blue'),
            ('avg_num_citations', 'Citations Count', 'green'),
            ('avg_lexical_diversity', 'Lexical Diversity', 'orange'),
            ('avg_query_answer_similarity', 'Query-Answer Similarity', 'red'),
            ('avg_avg_rouge1', 'ROUGE-1 Score', 'purple'),
            ('avg_avg_rougeL', 'ROUGE-L Score', 'brown')
        ]
        
        for idx, (metric_key, title, color) in enumerate(metrics):
            if metric_key in generation_data:
                ax = axes[idx]
                value = generation_data[metric_key]
                
                ax.bar([''], [value], color=color, alpha=0.7)
                ax.set_title(f'{title}\n{value:.3f}')
                if 'Similarity' in title or 'Score' in title:
                    ax.set_ylim(0, 1.1)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Generation Metrics Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('generation_metrics.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_performance_timeline(self):
        """Визуализация временных метрик"""
        queries = [f"Q{i+1}" for i in range(self.results['summary']['total_queries_evaluated'])]
        
        retrieval_times = []
        generation_times = []
        total_times = []
        
        for result in self.results['per_query_results']:
            retrieval_times.append(result['system_metrics']['retrieval_time_seconds'])
            generation_times.append(result['system_metrics']['generation_time_seconds'])
            total_times.append(result['system_metrics']['total_time_seconds'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(queries))
        width = 0.25
        
        ax.bar([i - width for i in x], retrieval_times, width, label='Retrieval', color='blue', alpha=0.7)
        ax.bar(x, generation_times, width, label='Generation', color='green', alpha=0.7)
        ax.bar([i + width for i in x], total_times, width, label='Total', color='red', alpha=0.7)
        
        ax.set_xlabel('Query Number')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Pipeline Performance Timeline')
        ax.set_xticks(x)
        ax.set_xticklabels(queries)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_timeline.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_quality_radar(self):
        """Радар-график качества ответов"""
        quality_data = self.results['average_metrics']['quality_metrics']
        
        metrics = [
            'avg_topic_coverage',
            'avg_has_citations',
            'avg_has_multiple_sentences',
            'avg_has_reasonable_length',
            'avg_quality_score'
        ]
        
        labels = ['Topic Coverage', 'Has Citations', 'Multiple Sentences', 
                 'Reasonable Length', 'Overall Quality']
        
        values = [quality_data.get(metric, 0) for metric in metrics]
        
        # Замыкаем круг
        values += values[:1]
        labels = labels + [labels[0]]
        
        angles = [n / float(len(labels) - 1) * 2 * 3.14159 for n in range(len(labels))]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax.fill(angles, values, alpha=0.25, color='blue')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels[:-1], fontsize=12)
        ax.set_ylim(0, 1)
        
        ax.set_title('Answer Quality Metrics Radar', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('quality_radar.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self):
        """Генерация комплексного отчета с визуализациями"""
        self.plot_retrieval_metrics()
        self.plot_generation_metrics()
        self.plot_performance_timeline()
        self.plot_quality_radar()
        
        # Создание текстового отчета
        report = f"""
        COMPREHENSIVE EVALUATION SUMMARY
        {'='*50}
        
        Overall Quality Score: {self.results['summary']['avg_quality_score']:.3f}/1.0
        
        Key Strengths:
        • Retrieval Similarity: {self.results['average_metrics']['retrieval_metrics'].get('avg_avg_similarity', 0):.3f}
        • Answer Relevance: {self.results['average_metrics']['generation_metrics'].get('avg_query_answer_similarity', 0):.3f}
        • Citation Rate: {self.results['average_metrics']['generation_metrics'].get('avg_num_citations', 0):.1f} per answer
        
        Areas for Improvement:
        • Response Time: {self.results['summary']['avg_total_time']:.2f}s (target: <3s)
        • Answer Length: {self.results['summary']['avg_answer_length']:.0f} words (target: >150)
        • ROUGE-L Score: {self.results['average_metrics']['generation_metrics'].get('avg_avg_rougeL', 0):.3f} (target: >0.4)
        
        Recommendations:
        1. Optimize retrieval for faster response times
        2. Improve prompt engineering for more comprehensive answers
        3. Implement caching for frequent queries
        4. Add confidence scoring for better source selection
        """
        
        print(report)
        
        # Сохранение отчета
        with open('evaluation_summary.txt', 'w') as f:
            f.write(report)

# Запуск визуализации
if __name__ == "__main__":
    visualizer = MetricsVisualizer('evaluation_results.json')
    visualizer.generate_comprehensive_report()