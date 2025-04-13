# 🎯 Targeted Adverse News Screening

A Streamlit-based web application for entity-focused monitoring of adverse financial news using Azure OpenAI and SerperAPI.

## 🚀 Features

- **🔍 Targeted News Search**  
  Search for news articles by entity and time range using SerperAPI.

- **🧠 Named Entity Recognition (NER)**  
  Automatically extract named entities from news headlines:
  ```
  COMPANY, PERSON, FINANCIAL_INSTITUTION, REGULATORY_BODY, POTENTIAL_CRIME,
  LEGAL_ACTION, ENFORCEMENT_ACTION, LOCATION, SANCTION_ENTITY, SECTOR, REGULATION
  ```

- **📂 Zero-shot News Classification**  
  Categorize news into pre-defined adverse risk types:
  ```
  Money Laundering, Terrorist Financing, Sanctions Violations, Fraud, Tax Evasion,
  Bribery and Corruption, Insider Trading, Ponzi and Pyramid Schemes,
  Trade-Based Money Laundering, General Financial News, Non Financial News
  ```

- **📈 Relevance Scoring**  
  Rule-based scoring that combines:
  - LLM classification confidence
  - Mention count of key entities
  - High-risk entity-category combinations

- **📊 Interactive Dashboards**
  - Distribution of news by risk category
  - Time trends of entity and category mentions
  - Entity drill-down with:
    - Top mentioned entities by type
    - Entity-category matrix with highlighted risks

- **📰 News Article View**
  - Title, snippet, publish date, URL, extracted entities

- **💾 Export & Drill-down**
  - Drill down by entity
  - Export results as CSV

## ⚙️ Architecture

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**:
  - [SerperAPI](https://serper.dev/) for news search
  - [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/) for classification and NER

## 📁 Repository Structure

```
Adverse-News-Screening
├── .gitignore
├── LICENSE
├── README.md
├── app.py                      # Streamlit frontend
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── adverse_relevance_scorer.py     # Rule-based relevance scoring
│   ├── llm_news_analyzer.py            # Azure OpenAI NER + classification
│   ├── news_collector.py               # SerperAPI-based news search
│   └── utils.py
└── tests/
    ├── __init__.py
    ├── test_adverse_relevance_scorer.py
    ├── test_news_analyzer.py
    └── test_news_collector.py
```


## 🧪 Setup & Run

### 1. Prerequisites
- ✔️ python 3.9
- [✔️ SerperAPI key](https://serper.dev/)
- [✔️ Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal)


### 2. Clone the repository

```bash
git clone https://github.com/xiaoman0220/Adverse-News-Screening.git
cd Adverse-News-Screening
```


### 3. Install dependencies

```bash
pip install -r requirements.txt
```


### 4. Run unit tests

```bash
cd tests
python -m unittest
cd ..
```


### 5. Set environment variables

```bash
export NEWS_SEARCH_KEY=<your-serperapi-key>
export OPENAI_ENDPOINT=<your-azure-endpoint>
export OPENAI_DEPLOYMENT_NAME=<your-deployment-name>
export OPENAI_SUBSCRIPTION_KEY=<your-openai-key>
export OPENAI_VERSION=<your-model-version>
```


### 6. Launch the app

```bash
streamlit run app.py
```


## 🌐 Deployment
The app is deployed on [Streamlit Cloud](https://xiaoman0220-adverse-news-screening-app-pi60gp.streamlit.app/)  
**Access is restricted to authorized users.**


## 📄 License

MIT License