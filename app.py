import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from src.utils import parse_date, patch_dataframe_date
from src.news_collector import NewsCollector
from src.llm_news_analyzer import LLMNewsAnalyzer
from src.adverse_relevance_scorer import AdverseRelevanceScorer

# --------------------------
# Page Configurations
# --------------------------
st.set_page_config(
    page_title="Targeted Adverse News Screening",
    layout="wide",
    page_icon="🔍"
)
st.markdown("""
    <style>
    .card {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 1.4rem !important;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Side bar
# --------------------------
with st.sidebar:
    st.title("🎯 Targeted Adverse News Screening")
    st.markdown("---")
    
    # Enter search query
    st.subheader("🔍 Search Criteria")
    query = st.text_input(
        "Entity Name", 
        placeholder="e.g. Wirecard, Sam Bankman-Fried",
        help="Enter the entity name you want to investigate"
    )
    
    time_range = st.selectbox(
        "Date Range",
        options=["Past week", 
                "Past month", "Past year", "Any time"],
        index=3,
        help="Select the time range for news collection"
    )
    
    return_num = st.slider(
        "Number of Articles",
        min_value=10, max_value=100, value=10, step=10,
        help="Adjust the number of articles to analyze"
    )
    
    if st.button("🚀 Start Analysis", use_container_width=True):
        if query != st.session_state.current_query or \
            time_range != st.session_state.search_time_range or \
            return_num != st.session_state.search_return_num:
            st.session_state.analysis_triggered = True
            st.session_state.current_query = query
            st.session_state.search_time_range = time_range
            st.session_state.search_return_num = return_num
            st.session_state.query_data = None
        

# --------------------------
# Fetch and process news data
# --------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news_data(_collector, query, return_num, time_range):
    """fetch news data"""
    _collector.search(query=query, return_num=return_num, time_range=time_range)
    return _collector.result

def process_news_data(articles):
    """process news data"""
    analyzer = LLMNewsAnalyzer()
    processed_data = []
    
    # batch processing
    batch_size = 10
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i+batch_size]
        
        # prepare batch articles
        batch_texts = [
            f"[title]{art['title']} [snippet]{art.get('snippet', '')}"
            for art in batch
        ]
        
        # classification
        analyzer.classify_news("\n".join(batch_texts))
        classification = analyzer.classification_result

        # ner
        analyzer.extract_entities("\n".join(batch_texts))
        entities = analyzer.ner_result

        # combine results
        for idx, art in enumerate(batch):
            # calculate relevance score
            scorer = AdverseRelevanceScorer(entities[idx], classification[idx]['confidence_score'])
            scorer.compute_relevant_score()

            processed_data.append({
                "title": art.get("title"),
                "snippet": art.get("snippet", ""),
                "url": art.get("link", "#"),
                "source": art.get("source"),
                "date": parse_date(art.get("date", "N/A")),
                "category": classification[idx]['category'],
                "score": classification[idx]['confidence_score'],
                "justification": classification[idx]['justification'],
                "entities": entities[idx],
                "adverse_relevance": scorer.relevance_score
            })
    
    return pd.DataFrame(processed_data)

# --------------------------
# Visualization
# --------------------------
def render_metrics(news_data):
    """render key metrics"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Articles", len(news_data))
    with col2:
        adverse_count = len(news_data[~news_data['category'].isin(['Non Financial News', 'General Financial News'])])
        st.metric("Adverse Findings", adverse_count, delta_color="off")
    with col3:
        latest_date = news_data['date'].max().strftime("%Y-%m-%d")
        st.metric("Latest Update", latest_date)

def render_entity_selection():
    entity_types = [
                    "COMPANY",
                    "PERSON",
                    "FINANCIAL_INSTITUTION",
                    "REGULATORY_BODY",
                    "PROTENTIAL_CRIME", 
                    "LEGAL_ACTION", 
                    "ENFORCEMENT_ACTION",
                    "LOCATION",
                    "SANCTION_ENTITY",
                    "SECTOR",
                    "REGULATION"
                ]
    entity_type_option = st.multiselect(
                "Select Entity Type",
                entity_types,
                ["PERSON", "COMPANY"],
            )
    return entity_type_option

def render_category_distribution(news_data):
    """Render category distribution"""
    st.subheader("📰 News Category Distribution")
    fig = px.pie(news_data, names='category', 
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)


def render_category_entity_matrix(news_data, entity_type_options):
    """Render category entity matrix"""
    matrix_records = []
    for _, row in news_data.iterrows():
        category = row["category"]
        entities_in_row = {}
        for entity_type, ents in row["entities"].items():
            if entity_type not in  entity_type_options:
                continue
            if isinstance(ents, list):
                for ent in ents:
                    if isinstance(ent, dict) and "entity_name" in ent:
                        entities_in_row[ent["entity_name"]] = entity_type
                    elif isinstance(ent, str):
                        entities_in_row.append(ent)
        for ent, type in entities_in_row.items():
            matrix_records.append({"entity": ent, "entity_type": type, "category": category})
    if matrix_records:
        df_matrix = pd.DataFrame(matrix_records)
        pivot = pd.pivot_table(df_matrix, index=["entity", "entity_type"], columns=["category"], aggfunc=len, fill_value=0)
        def highlight_except(col):
            if col.name in ["General Financial News", "Non Financial News"]:
                return ["" for _ in col]
            else:
                return [f"background-color: rgba(255, 0, 0, {min(v/10,1)})" if v > 0 else "" for v in col]
        styled_pivot = pivot.style.apply(highlight_except, axis=0)
        st.write(styled_pivot)

def render_top_mentioned_entities(news_data, entity_type_options):
    """Render top mentioned entities"""
    entity_counter = Counter()
    for _, row in news_data.iterrows():
        for ent_type in entity_type_options:
            for ent in row['entities'].get(ent_type, []):
                if isinstance(ent, dict):
                    entity_counter[ent["entity_name"]] += 1

    if entity_counter:
        top_entities = pd.DataFrame(entity_counter.most_common(10), columns=["Entity", "Count"])
        fig_entities = px.bar(top_entities, x="Entity", y="Count", title=f"Top Mentioned Entities")
        st.plotly_chart(fig_entities)
    else:
        st.markdown("_No mention_")

def render_time_trend(news_data):
    """Render time trend"""
    st.subheader(f"📈 News Trend Related to {query}")
    col1, col2 = st.columns(2)
    with col1:
        if not news_data.empty:
            category_trend = news_data.groupby(by=["date", "category"]).size().reset_index(name="article_count")
            df_full = patch_dataframe_date(category_trend, "category")
            fig = px.line(df_full, x='date', y='article_count', color='category', title="Time Series Trend by News Category")
            st.plotly_chart(fig)
    with col2:
        timeline_records = []
        for _, row in news_data.iterrows():
            try:
                dt = pd.to_datetime(row["date"], errors="coerce")
                if pd.isna(dt):
                    continue
            except Exception:
                continue
            mentioned_entities = []
            for ents in row["entities"].values():
                if isinstance(ents, list):
                    for ent in ents:
                        if isinstance(ent, dict) and "entity_name" in ent:
                            mentioned_entities.append(ent["entity_name"])
                        elif isinstance(ent, str):
                            mentioned_entities.append(ent)
            for ent in set(mentioned_entities):
                timeline_records.append({"date": dt, "entity": ent})
        if timeline_records:
            entity_trend = pd.DataFrame(timeline_records)
            entity_trend = entity_trend.groupby(["date", "entity"]).size().reset_index(name="count")
            df_full = patch_dataframe_date(entity_trend, "entity")
            fig_timeline = px.line(df_full, x="date", y="count", color="entity", title="Time Series Trend by Entities")
            st.plotly_chart(fig_timeline)

# --------------------------
# Display article details
# --------------------------
def display_article(row):
    # Display news articles
    is_adverse = row['category'] not in ['General Financial News', 'Non Financial News']
    news_tag = "🔴" if is_adverse else "🔵"
    # for _, row in df.iterrows():
    with st.expander(f"{news_tag} {row['title']}"):
        st.markdown(f"**Published**: {row['date']}")
        st.markdown(f"**URL**: [Link]({row['url']})")
        st.markdown(f"**Snippet**: {row['snippet']}")
        if is_adverse:
            st.markdown(f"**Predicted Category**: :red-background[{row['category']}]")
        else:
            st.markdown(f"**Predicted Category**: :blue-background[{row['category']}]")
        st.markdown(f"**Justification**: `{row['justification']}`")
        if is_adverse:
            st.markdown(f"**📌Adverse Relevance**: :red[{row['adverse_relevance']}]")

        st.markdown("**Entities Extracted:**")
        entities = row['entities']
        entities_flag = 0
        for entity_type, entity_list in entities.items():
            if not entity_list:
                continue
            entities_flag = 1
            st.markdown(f"**🧠 {entity_type.replace('_', ' ').title()}**")
            for entity in entity_list:
                if isinstance(entity, dict):
                    name = entity.get("entity_name")
                    variations = entity.get("variations", [])
                    variation_str = ", ".join([v for v in variations if v.lower() != name.lower()])
                    if variation_str:
                        st.markdown(f"- **{name}**  \
                        _Also referred to as:_ {variation_str}")
                    else:
                        st.markdown(f"- **{name}** \
                                    _No other reference found_")
        if not entities_flag:
            st.markdown("_No entities found_")

# --------------------------
# Main page
# --------------------------
def main():
    
    if 'analysis_triggered' not in st.session_state:
        st.session_state.analysis_triggered = False
    if "current_query" not in st.session_state:
        st.session_state.current_query = None
    if "search_time_range" not in st.session_state:
        st.session_state.search_time_range = ""
    if "search_return_num" not in st.session_state:
        st.session_state.search_return_num = 0
    if "query_data" not in st.session_state:
        st.session_state.query_data = None
    
    if st.session_state.analysis_triggered and (st.session_state.current_query != query or \
                                                st.session_state.search_time_range != time_range or \
                                                st.session_state.search_return_num != return_num or \
                                                st.session_state.query_data is None):
        with st.spinner("🔍 Collecting and analyzing news articles..."):
            collector = NewsCollector()
            raw_data = fetch_news_data(collector, 
                                     st.session_state.current_query,
                                     st.session_state.search_return_num, 
                                     st.session_state.search_time_range)
            if not raw_data or "news" not in raw_data:
                st.error("No relevant news found")
                return
            
            
            news_df = process_news_data(raw_data["news"])
            news_df['date'] = pd.to_datetime(news_df['date'], errors="coerce")
            st.session_state.query_data = news_df

        st.session_state.analysis_triggered = False

    if st.session_state.query_data is not None:
        
        news_df = st.session_state.query_data
        # Main page layout
        tab1, tab2 = st.tabs(["Analysis Dashboard", "Article Details"])
        
        with tab1:
            render_metrics(news_df)
            st.markdown("---")
            
            render_category_distribution(news_df)
            
            render_time_trend(news_df)
            
            st.subheader("🏷️ Entities Drill Down")
            entity_type_options = render_entity_selection()
            col1, col2 = st.columns(2)
            with col1:
                render_top_mentioned_entities(news_df, entity_type_options)
            with col2:
                render_category_entity_matrix(news_df, entity_type_options)
                
        with tab2:
            # News article result layout
            st.subheader("📑 Article Review Panel")
            news_df = news_df.sort_values(by="date", ascending=False)
            csv_data = news_df.drop(columns=["entities", "justification", "score"]).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="news_data.csv",
                mime="text/csv",
                key="download_csv"
                )
            
            # Pagination
            page_size = 10
            page_number = st.number_input('Page', min_value=1, 
                                        max_value=len(news_df)//page_size+1,
                                        key='article_pager')
            
            for idx in range((page_number-1)*page_size, page_number*page_size):
                if idx >= len(news_df):
                    break
                article = news_df.iloc[idx]
                
                display_article(article)

if __name__ == "__main__":
    main()