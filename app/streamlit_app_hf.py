# app/streamlit_app_hf.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Добавление пути к src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.embeddings import EmbeddingManagerHF
from src.generation_hf import HFResponseGenerator
from src.rag_pipeline_hf import RAGPipelineHF

# Конфигурация страницы
st.set_page_config(
    page_title="Alzheimer's RAG Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация состояния сессии
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_response' not in st.session_state:
    st.session_state.current_response = None

@st.cache_resource
def initialize_rag_system():
    """Инициализация RAG системы с кэшированием"""
    st.info("🚀 Initializing RAG System...")
    
    # Выбор модели в зависимости от доступных ресурсов
    import torch
    has_gpu = torch.cuda.is_available()
    
    if has_gpu:
        embedding_model = "BAAI/bge-base-en-v1.5"
        generation_model = "microsoft/phi-2"
        st.success(f"GPU detected! Using {generation_model}")
    else:
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        generation_model = "google/flan-t5-base"
        st.info(f"Using CPU-optimized models: {generation_model}")
    
    # Инициализация компонентов
    embedding_manager = EmbeddingManagerHF(model_name=embedding_model)
    response_generator = HFResponseGenerator(model_name=generation_model)
    rag_pipeline = RAGPipelineHF(embedding_manager, response_generator)
    
    return rag_pipeline

def main():
    # Заголовок
    st.title("🧬 Alzheimer's Disease Research Assistant")
    st.markdown("""
    ### RAG System for Drug Target Discovery
    This AI assistant helps researchers find potential drug targets for Alzheimer's disease 
    by searching through scientific literature and providing evidence-based answers.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Выбор количества источников
        n_sources = st.slider(
            "Number of sources to retrieve",
            min_value=3,
            max_value=10,
            value=5,
            help="How many research excerpts to use for generating answers"
        )
        
        # Порог релевантности
        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Minimum relevance score for sources"
        )
        
        # Длина ответа
        max_tokens = st.slider(
            "Maximum answer length",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="Maximum tokens in generated answer"
        )
        
        # Примеры запросов
        st.header("📋 Example Queries")
        example_queries = [
            "What are the most promising tau protein targets for Alzheimer's disease?",
            "Are there any small molecule inhibitors targeting amyloid beta aggregation?",
            "What immunotherapies are being developed for Alzheimer's disease?",
            "How does neuroinflammation contribute to Alzheimer's progression and what are the therapeutic targets?",
            "What are the latest developments in targeting APOE4 for Alzheimer's treatment?"
        ]
        
        for query in example_queries:
            if st.button(f"`{query[:50]}...`", key=query):
                st.session_state.query_input = query
        
        # Статистика
        st.header("📊 Statistics")
        if st.session_state.rag_pipeline:
            st.metric("Articles in database", "50+")
            st.metric("Chunks indexed", "500+")
            st.metric("Embedding model", "BGE-base")
            st.metric("Generation model", "Phi-2")
    
    # Основная область
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Инициализация системы
        if st.session_state.rag_pipeline is None:
            if st.button("Initialize RAG System", type="primary"):
                with st.spinner("Loading models... This may take a few minutes."):
                    st.session_state.rag_pipeline = initialize_rag_system()
                    st.success("System initialized successfully!")
                    st.rerun()
        else:
            # Поле для запроса
            query = st.text_area(
                "💭 **Enter your research question:**",
                height=120,
                placeholder="e.g., What are the latest targets for reducing tau phosphorylation in Alzheimer's disease?",
                value=st.session_state.get('query_input', '')
            )
            
            col1_1, col1_2 = st.columns([3, 1])
            with col1_1:
                ask_button = st.button("🔍 Search & Generate Answer", type="primary", use_container_width=True)
            with col1_2:
                clear_button = st.button("🗑️ Clear", use_container_width=True)
            
            if clear_button:
                st.session_state.current_response = None
                st.session_state.query_input = ""
                st.rerun()
            
            if ask_button and query:
                with st.spinner("Searching through research literature..."):
                    # Выполнение запроса
                    response = st.session_state.rag_pipeline.query(
                        question=query,
                        n_sources=n_sources,
                        threshold=similarity_threshold
                    )
                    
                    # Сохранение в истории
                    st.session_state.query_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": query,
                        "confidence": response["confidence"]
                    })
                    
                    st.session_state.current_response = response
    
    with col2:
        st.info("### 💡 Tips")
        st.markdown("""
        - Be specific in your questions
        - Ask about mechanisms, targets, or therapies
        - The system uses 50+ recent Alzheimer's research articles
        - Answers are generated based on retrieved sources
        - Always verify with original papers
        """)
        
        # История запросов
        if st.session_state.query_history:
            st.subheader("📜 Recent Queries")
            for i, item in enumerate(st.session_state.query_history[-5:]):
                st.caption(f"{item['timestamp'].split()[1]} - {item['query'][:40]}...")
    
    # Отображение ответа
    if st.session_state.current_response:
        response = st.session_state.current_response
        
        # Визуализация уверенности
        col_confidence, col_sources = st.columns([1, 1])
        
        with col_confidence:
            st.metric("Confidence Score", f"{response['confidence']:.2%}")
            
            # График уверенности
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=response['confidence'] * 100,
                title={'text': "Answer Confidence"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_sources:
            st.metric("Sources Used", len(response['sources']))
            
            # Визуализация источников по годам
            years = []
            for source in response['sources']:
                if 'year' in source and source['year'].isdigit():
                    years.append(int(source['year']))
            
            if years:
                year_counts = pd.Series(years).value_counts().sort_index()
                fig2 = px.bar(
                    x=year_counts.index,
                    y=year_counts.values,
                    labels={'x': 'Publication Year', 'y': 'Number of Sources'},
                    title="Sources by Publication Year"
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        # Отображение ответа
        st.markdown("---")
        st.subheader("📝 Generated Answer")
        
        # Красивое оформление ответа
        answer_container = st.container()
        with answer_container:
            st.markdown(f"""
            <div style='
                background-color: #f0f7ff;
                border-radius: 10px;
                padding: 20px;
                border-left: 5px solid #4a90e2;
                margin-bottom: 20px;
            '>
            {response['answer']}
            </div>
            """, unsafe_allow_html=True)
        
        # Отображение источников
        st.subheader("📚 Source Documents")
        
        for i, source in enumerate(response['sources']):
            with st.expander(f"Source {i+1}: {source['title'][:70]}...", expanded=(i < 2)):
                col_left, col_right = st.columns([3, 1])
                
                with col_left:
                    st.markdown(f"**Excerpt:** {source['excerpt']}")
                    
                    if source.get('authors'):
                        st.caption(f"**Authors:** {', '.join(source['authors'][:3])}")
                    
                    st.caption(f"**Journal:** {source['journal']} ({source.get('year', 'N/A')})")
                
                with col_right:
                    # Иконки действий
                    if source['url']:
                        st.link_button("📄 Original", source['url'])
                    
                    if source['doi']:
                        st.code(source['doi'], language=None)
                    
                    st.metric("Relevance", f"{source['relevance_score']:.2%}")
                    
                    if source.get('cited'):
                        st.success("Cited in answer ✓")
                    else:
                        st.warning("Not cited in answer")
        
        # Кнопки действий
        col_actions = st.columns(4)
        with col_actions[0]:
            if st.button("📥 Export Answer"):
                # Экспорт ответа
                export_data = {
                    "question": response['question'],
                    "answer": response['answer'],
                    "sources": response['sources'],
                    "timestamp": datetime.now().isoformat()
                }
                st.download_button(
                    label="Download as JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"alzheimer_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col_actions[1]:
            if st.button("🔄 Generate Alternative"):
                # Регенерация ответа с другими параметрами
                pass
        
        with col_actions[2]:
            if st.button("📊 Analyze Targets"):
                # Анализ упомянутых мишеней
                targets = extract_targets_from_response(response)
                if targets:
                    st.write("**Identified Targets:**", ", ".join(targets))
        
        with col_actions[3]:
            if st.button("❓ Ask Follow-up"):
                # Подсказка для follow-up вопроса
                follow_up = generate_follow_up(response['question'])
                st.text_area("Suggested follow-up:", follow_up)

def extract_targets_from_response(response):
    """Извлечение упомянутых мишеней из ответа"""
    import re
    
    answer = response['answer'].lower()
    
    # Список известных мишеней при болезни Альцгеймера
    known_targets = [
        'tau', 'amyloid', 'beta-amyloid', 'aβ', 'apoe', 'apoe4',
        'bace1', 'bace', 'gsk3β', 'gsk3', 'cdk5', 'ppp',
        'trem2', 'cd33', 'app', 'psen1', 'psen2',
        'neprilysin', 'ide', 'ace', 'nrf2', 'nf-κb'
    ]
    
    found_targets = []
    for target in known_targets:
        if target in answer:
            found_targets.append(target)
    
    return list(set(found_targets))

def generate_follow_up(original_question):
    """Генерация предложения для follow-up вопроса"""
    follow_ups = [
        "What are the clinical trial results for these targets?",
        "Are there any safety concerns with targeting this pathway?",
        "What biomarkers are associated with these targets?",
        "How do these targets interact with each other?",
        "What are the latest drug candidates targeting this mechanism?"
    ]
    
    # Простая эвристика для выбора follow-up
    if 'tau' in original_question.lower():
        return "What are the latest tau PET imaging biomarkers?"
    elif 'amyloid' in original_question.lower():
        return "How do current amyloid-targeting therapies perform in clinical trials?"
    elif 'neuroinflammation' in original_question.lower():
        return "What microglial targets are being investigated?"
    
    return follow_ups[0]

if __name__ == "__main__":
    main()