import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList
)
from typing import List, Dict, Any
import re

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class HFResponseGenerator:
    def __init__(self, model_name="microsoft/phi-2", device_map="auto"):
        """
        Инициализация генеративной модели
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading generation model on {self.device}...")
        
        # Определяем тип модели по имени
        if "t5" in model_name.lower():
            self.model_type = "seq2seq"
            self.model_class = AutoModelForSeq2SeqLM
        else:
            self.model_type = "causal"
            self.model_class = AutoModelForCausalLM
        
        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Добавление pad token если его нет
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Определение параметров загрузки модели
        load_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "trust_remote_code": True,
        }
        
        # Только для GPU используем device_map, для CPU - нет
        if self.device == "cuda":
            load_kwargs["device_map"] = device_map
        else:
            # Для CPU не используем device_map
            load_kwargs["device_map"] = None
        
        # Для CPU используем float32
        if self.device == "cpu":
            load_kwargs["torch_dtype"] = torch.float32
        
        # Загрузка модели
        try:
            self.model = self.model_class.from_pretrained(
                model_name,
                **load_kwargs
            )
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            # Fallback to distilgpt2
            print("Falling back to distilgpt2...")
            model_name = "distilgpt2"
            self.model_type = "causal"
            self.model_class = AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            load_kwargs["torch_dtype"] = torch.float32
            load_kwargs["device_map"] = None
            self.model = self.model_class.from_pretrained(
                model_name,
                **load_kwargs
            )
        
        # Если модель не загружена на GPU, перемещаем вручную
        if self.device == "cuda" and not next(self.model.parameters()).is_cuda:
            self.model = self.model.cuda()
        
        # Создание pipeline БЕЗ указания device
        if self.model_type == "seq2seq":
            self.generator = pipeline(
                "text2text-generation",  # Для T5 используем text2text-generation
                model=self.model,
                tokenizer=self.tokenizer,
                # Не указываем device здесь!
            )
        else:
            self.generator = pipeline(
                "text-generation",  # Для GPT-like моделей
                model=self.model,
                tokenizer=self.tokenizer,
                # Не указываем device здесь!
            )
        
        # Параметры генерации
        self.generation_config = {
            "max_new_tokens": 800,  # Увеличим для более детальных ответов
            "temperature": 0.3,     # Снизим для более точных ответов
            "top_p": 0.9,
            "top_k": 40,
            "do_sample": True,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
    
    def format_prompt(self, query: str, contexts: List[str], 
                     metadatas: List[Dict]) -> str:
        """Форматирование промпта для модели с включением кратких содержаний"""
        
        # Создание строки с контекстом и кратким содержанием
        context_str = ""
        for i, (context, metadata) in enumerate(zip(contexts, metadatas), 1):
            # Получаем краткое содержание из метаданных или создаем из контекста
            summary = metadata.get('summary', '')
            if not summary and 'abstract' in metadata:
                summary = metadata['abstract']
            
            # Если нет summary/abstract, создаем краткое из контекста
            if not summary:
                # Берем первые 2 предложения как summary
                sentences = context.split('. ')
                summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else context[:200] + "..."
            
            source_info = f"[Source {i}]"
            if 'title' in metadata:
                source_info += f" Title: {metadata['title']}"
            if 'year' in metadata:
                source_info += f" ({metadata['year']})"
            
            context_str += f"\n\n{'='*80}\n{source_info}\n{'-'*80}"
            context_str += f"\n📝 Article Summary: {summary[:300]}..."
            context_str += f"\n📄 Relevant Excerpt: {context[:500]}..."
        
        # Улучшенный системный промпт
        system_prompt = """You are an expert biomedical research assistant specializing in Alzheimer's disease. 
Your task is to provide a comprehensive, evidence-based answer using ONLY the provided research excerpts.

IMPORTANT INSTRUCTIONS:
1. Start with a clear, concise summary answer
2. For each key point, cite the specific source using [Source X] notation
3. If multiple sources support a claim, cite all relevant sources [Source X, Source Y]
4. If information is missing from provided sources, explicitly state this
5. Structure your answer logically with clear paragraphs
6. Include a brief conclusion summarizing the key findings

CRITICAL: Do not make up any information not present in the provided sources."""
        
        # Разный формат для разных типов моделей
        if self.model_type == "seq2seq":
            # Для T5
            prompt = f"""Based on the following research article summaries and excerpts, answer the query:

Query: {query}

Research Sources:{context_str}

Provide a detailed, evidence-based answer that cites specific sources:"""
        else:
            # Для GPT-like моделей
            prompt = f"""{system_prompt}

RESEARCH QUESTION: {query}

RESEARCH SOURCES (with summaries and relevant excerpts):{context_str}

YOUR TASK: Based ONLY on the above research sources, provide a comprehensive answer that:
1. Answers the research question
2. Cites specific sources for each claim
3. Summarizes key findings from each relevant source

ANSWER STRUCTURE:
- Brief overall summary
- Detailed analysis with source citations
- Conclusion with key takeaways

BEGIN ANSWER:"""
        
        return prompt
    
    def generate_answer(self, query: str, contexts: List[str], 
                       metadatas: List[Dict]) -> Dict[str, Any]:
        """Генерация ответа на основе контекста"""
        
        if not contexts:
            return {
                "answer": "No relevant research context provided to answer this question.",
                "prompt_used": "",
                "model_type": self.model_type,
                "success": False
            }
        
        # Форматирование промпта
        prompt = self.format_prompt(query, contexts, metadatas)
        
        print(f"Generated prompt length: {len(prompt)} characters")
        
        # Генерация
        try:
            # Используем генератор
            outputs = self.generator(
                prompt,
                **self.generation_config
            )
            
            generated_text = outputs[0]['generated_text']
            
            # Для causal моделей удаляем промпт из начала ответа
            if self.model_type == "causal" and generated_text.startswith(prompt):
                answer = generated_text[len(prompt):].strip()
            else:
                answer = generated_text.strip()
            
            print(f"Generated answer length: {len(answer)} characters")
            
            # Пост-обработка ответа
            answer = self.postprocess_answer(answer, contexts, metadatas)
            
            # Проверка качества ответа
            if len(answer) < 100:
                print("Answer too short, enriching with source information...")
                answer = self.enrich_short_answer(answer, contexts, metadatas)
            
            return {
                "answer": answer,
                "prompt_used": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                "model_type": self.model_type,
                "success": True
            }
            
        except Exception as e:
            print(f"Error in generation: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback ответ с источниками
            fallback_answer = self.create_fallback_answer(query, contexts, metadatas)
            
            return {
                "answer": fallback_answer,
                "prompt_used": prompt[:500] + "...",
                "model_type": self.model_type,
                "success": False,
                "error": str(e)
            }
    
    def postprocess_answer(self, answer: str, contexts: List[str], 
                          metadatas: List[Dict]) -> str:
        """Пост-обработка сгенерированного ответа"""
        
        # Удаление повторяющихся предложений
        sentences = answer.split('. ')
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        answer = '. '.join(unique_sentences)
        
        # Добавление разделителя
        answer = answer.strip()
        
        return answer
    
    def enrich_short_answer(self, answer: str, contexts: List[str], 
                           metadatas: List[Dict]) -> str:
        """Обогащение короткого ответа информацией из источников"""
        
        enriched = f"{answer}\n\n"
        enriched += "🔍 **Additional Information from Sources:**\n\n"
        
        for i, (context, metadata) in enumerate(zip(contexts, metadatas), 1):
            title = metadata.get('title', f"Source {i}")
            authors = metadata.get('authors', 'Unknown authors')
            year = metadata.get('year', 'Unknown year')
            
            # Создаем краткое содержание
            summary = metadata.get('summary', '')
            if not summary:
                sentences = context.split('. ')
                summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else context[:150] + "..."
            
            enriched += f"**[{i}] {title}** ({year}, {authors})\n"
            enriched += f"📝 *Summary:* {summary}\n\n"
        
        return enriched
    
    def create_fallback_answer(self, query: str, contexts: List[str], 
                              metadatas: List[Dict]) -> str:
        """Создание fallback ответа на основе источников"""
        
        answer = f"Based on the {len(contexts)} relevant research articles, here's what I found for your query about '{query}':\n\n"
        
        for i, (context, metadata) in enumerate(zip(contexts, metadatas), 1):
            title = metadata.get('title', f"Article {i}")
            year = metadata.get('year', '')
            
            # Извлекаем ключевую информацию
            key_points = []
            
            # Проверяем наличие ключевых слов в контексте
            keywords = ['inhibitors', 'treatment', 'therapy', 'target', 'mechanism', 'effect']
            sentences = context.split('. ')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in keywords):
                    key_points.append(sentence.strip())
            
            if key_points:
                answer += f"**Source {i}: {title}** {f'({year})' if year else ''}\n"
                answer += f"• {key_points[0]}\n"
                if len(key_points) > 1:
                    answer += f"• {key_points[1]}\n"
                answer += "\n"
        
        answer += "\n⚠️ *Note: This is an automatically generated summary based on available research excerpts.*"
        
        return answer