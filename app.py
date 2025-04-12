import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
import matplotlib.pyplot as plt

from collections import Counter
from datetime import date, timedelta
from wordcloud import WordCloud

from src.utils import parse_date, display_article, patch_dataframe_date
from src.news_collector import NewsCollector
from src.llm_news_analyzer import LLMNewsAnalyzer
from src.adverse_relevance_scorer import AdverseRelevanceScorer


st.set_page_config(page_title="Targeted Adverse News Screening", layout="wide")
alt.themes.enable("dark")

if "search_triggered" not in st.session_state:
    st.session_state.search_triggered = False
if "query_data" not in st.session_state:
    st.session_state.query_data = None
if "time_range" not in st.session_state:
    st.session_state.time_range = "Past month"
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

#######################
# Sidebar
with st.sidebar:
    st.title("🎯 Targeted Adverse News Screening")

    st.subheader("🔍 Search News")
    query = st.text_input("Enter entity name", placeholder="e.g. Wirecard, Sam Bankman-Fried")
    time_range = st.selectbox("Date Range", 
                                options=["Past month",
                                        "Past hour",
                                        "Past 24 hours",
                                        "Past week",
                                        "Past year",
                                        "Any time"],
                                index=["Past month",
                                        "Past hour",
                                        "Past 24 hours",
                                        "Past week",
                                        "Past year",
                                        "Any time"].index(st.session_state.time_range),
                                key="time_range")
    

    return_num = st.slider("Number of articles to return", min_value=10, max_value=100, step=10)
    

    if st.button("Search"):
        if query != st.session_state.last_query:
            st.session_state.query_data = None
            st.session_state.last_query = query

#######################
# Search news and analyze
if st.session_state.query_data is None:
    
    if not query:
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching news and analyzing..."):

            # 1. Search News
            collector = NewsCollector()
            collector.search(query=query, return_num=return_num, time_range=time_range)
            # with open('test_search_result.json', 'r') as f:
            #     collector.result = json.load(f)

            if not collector.result or "news" not in collector.result or collector.result['news'] == []:
                st.error("No results found.")
            
            else:
                articles = collector.result["news"]
                analyzer = LLMNewsAnalyzer()
                all_data = []
                display_num = 0
                for article in articles:  
                    title = article["title"]
                    snippet = article.get("snippet", "")
                    date_published = article.get("date", "N/A")
                    url = article.get("link", "#")
                    source = article.get("source")
                    all_data.append({
                        "title": title,
                        "date": date_published,
                        "url": url,
                        "snippet": snippet
                    })

                news_data = pd.DataFrame(all_data)
                news_data['date'] = news_data['date'].apply(lambda x: parse_date(x))

                # Batch process data
                num_splits = max(len(news_data) // 10, 1)
                splits = np.array_split(news_data, num_splits)
                categories = []
                scores = []
                cat_reason = []
                all_entities = []
                for split in splits:
                    split['text'] = split.apply(lambda x: "[title]" + x["title"].replace("\n", " ").strip() + " [snippet]" + x['snippet'].replace("\n", " ").strip(), axis=1)
                    llm_query = '\n'.join(split['text'].tolist())

                    # Classify news
                    analyzer.classify_news(llm_query)
                    classification = analyzer.classification_result
                    categories.extend([c['category'] for c in classification])
                    scores.extend([c['confidence_score'] for c in classification])
                    cat_reason.extend([c['justification'] for c in classification])

                    # Extract entities
                    analyzer.extract_entities(llm_query)
                    all_entities.extend(analyzer.ner_result)

                news_data['category'] = categories
                news_data['score'] = scores
                news_data['justification'] = cat_reason
                news_data['entities'] = all_entities

                st.session_state.query_data = {
                                    "query": query,
                                    "dataframe": news_data,
                                    "articles": all_data,
                                    "entities": all_entities
                                }
                
#######################
# Visualization
if st.session_state.query_data:

    news_data = st.session_state.query_data["dataframe"]
    all_data = st.session_state.query_data["articles"]
    all_entities = st.session_state.query_data["entities"]

    news_data['date'] = pd.to_datetime(news_data['date']).dt.date

    col = st.columns(2, gap='medium')
    with col[0]:
        tabs = st.tabs(["Distribution", "Time Series"])
        with tabs[0]:
            # Entity Category Matrix
            st.subheader("🧩 Entity-Category Matrix")
            matrix_records = []
            for _, row in news_data.iterrows():
                category = row["category"]
                entities_in_row = {}
                for entity_type, ents in row["entities"].items():
                    if entity_type not in ["COMPANY", "PERSON"]:
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

            # Category distribution
            st.subheader("📊 Category Distribution")
            if not news_data.empty:
                fig_cat = px.histogram(news_data, x='category', title='Distribution of News Categories')
                st.plotly_chart(fig_cat)

            # Interactive entity bar chart
            st.subheader("🏷️ Top Mentioned Entities")
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
            entity_type_option = st.selectbox("Select Entity Type", 
                                            options=entity_types)
            
            
            entity_counter = Counter()
            for entities in all_entities:
                for ent in entities.get(entity_type_option, []):
                    if isinstance(ent, dict):
                        entity_counter[ent["entity_name"]] += 1

            if entity_counter:
                top_entities = pd.DataFrame(entity_counter.most_common(10), columns=["Entity", "Count"])
                fig_entities = px.bar(top_entities, x="Entity", y="Count", title=f"Top {entity_type_option.title()} Entities")
                st.plotly_chart(fig_entities)
            else:
                st.markdown("_No mention_")

        with tabs[1]:
            # Time trend visualization
            st.subheader(f"📈 News Trend Related to {query}")
            if not news_data.empty:
                category_trend = news_data.groupby(by=["date", "category"]).size().reset_index(name="article_count")
                df_full = patch_dataframe_date(category_trend, "category")
                fig = px.line(df_full, x='date', y='article_count', color='category', title="Time Series Trend by News Category")
                st.plotly_chart(fig)
            
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
            
                        
        
                                
    with col[1]:
        adverse_df = news_data[~news_data['category'].isin(['Non Financial News', 'General Financial News'])]
        if not adverse_df.empty:
            with st.container(height=500):
                st.subheader("🚩 Adverse News Article Results")
                
                relevance = []
                # Calculate relevance score
                for _, row in adverse_df.iterrows():
                    scorer = AdverseRelevanceScorer(row["entities"], row["score"])
                    scorer.compute_relevant_score()
                    relevance.append(scorer.relevance_score)
                adverse_df['adverse_relevance'] = relevance

                
                col1, col2 = st.columns(2)
                with col1:
                    ranking = st.radio(
                        "Rank by",
                        ["Relevance", "Publish Date"],
                        index=None,
                    )
                if ranking == 'Relevance':
                    adverse_df = adverse_df.sort_values(by=['adverse_relevance'], ascending=False)
                if ranking == 'Publish Date':
                    adverse_df = adverse_df.sort_values(by=['date'], ascending=False)
                with col2:
                    csv_data = adverse_df[["title", "date", "url", "snippet", "category", "entities", "adverse_relevance"]].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="news_data.csv",
                        mime="text/csv",
                        key="download_adverse_csv"
                    )
                display_article(adverse_df)

        others_df = news_data[news_data['category'].isin(['Non Financial News', 'General Financial News'])]
        if not others_df.empty:
            with st.container(height=500):
                st.subheader("📄 Other News Article Results")
                
                # rank non adverse news by publish date
                others_df = others_df.sort_values(by=['date'], ascending=False)
                csv_data = others_df[["title", "date", "url", "snippet", "category", "entities"]].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="news_data.csv",
                    mime="text/csv",
                    key="download_others_csv"
                )
                display_article(others_df, is_adverse=False)

        
