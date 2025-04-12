import os
import json
from openai import AzureOpenAI

class LLMNewsAnalyzer:
    '''
    A class that calls Azure OpenAI for financial news classification and entity extraction.
    '''
    def __init__(self, endpoint=None, deployment_name=None, api_key=None, api_version=None):
        self.ner_result = None
        self.classification_result = None
        
        # Set Azure OpenAI API configurations
        self.endpoint = endpoint if endpoint else os.environ["OPENAI_ENDPOINT"]
        self.deployment_name = deployment_name if deployment_name else os.environ["OPENAI_DEPLOYMENT_NAME"]
        self.api_key = api_key if api_key else os.environ["OPENAI_SUBSCRIPTION_KEY"]
        self.api_version = api_version if api_version else os.environ["OPENAI_VERSION"]
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
        )


    def extract_entities(self, news_text):
        messages = [
                {
                    "role": "system",
                    "content":
                    """
                    You are an expert in extracting named entities from financial news. I will be providing a set of entities and some news articles containing [title] and [snippet], and you will need to extract the entities from the articles. Multiple news articles will be seperated by a newline character.
                    
                    ## Constraints
                    1. References to the same entity should be grouped together while all variations should be listed.
                    2. For each article, the output should be in a dictionary. The dictionary should contain the entities identified from that article, and the fields should include each entity type, entity_name and variations.
                    3. Return the exact same number of dictionaries as the number of articles nested in a list. The order of the results should follow the order of the articles.
                    4. The output should be a JSON object. Only return the JSON object, nothing else. Do not include any other strings whatsoever, even it's for formatting.

                    ## Sample Output
                    {
                        "COMPANY": [
                            {
                            "entity_name": "...",
                            "variations": [
                                "...",
                            ]
                            },
                            ...
                        ],
                        "PERSON": [
                            {
                            "entity_name": "...",
                            "variations": [
                                "...",
                            ]
                            },
                            ...
                        ],
                        "FINANCIAL_INSTITUTION": [...],
                        "REGULATORY_BODY": [...],
                        "PROTENTIAL_CRIME": [...],
                        "LEGAL_ACTION": [...],
                        "ENFORCEMENT_ACTION": [...],
                        "LOCATION": [...],
                        "SANCTION_ENTITY": [...],
                        "SECTOR": [...],
                        "REGULATION": [...]
                        }
                    
                    ## Entities
                    COMPANY: Public or private companies, including subsidiaries.
                    PERSON: Individuals, especially executives, board members, or other key figures.
                    FINANCIAL_INSTITUTION: Banks, investment firms, hedge funds, etc.
                    REGULATORY_BODY: Agencies like SEC, MAS, FCA, etc.
                    PROTENTIAL_CRIME: Terms indicating potential crime.
                    LEGAL_ACTION: Mentions of lawsuits, indictments, probes, settlements.
                    ENFORCEMENT_ACTION: Actions like fines, bans, license suspensions.
                    LOCATION: Cities, countries—important for sanctions or jurisdiction context.
                    SANCTION_ENTITY: Country or individual sanctioned or under restrictions.
                    SECTOR: Industry or economic sector (e.g., "energy", "fintech").
                    REGULATION: Names or identifiers of laws or policies (e.g., GDPR, FCPA, Dodd-Frank).
                    """
                },
                {
                    "role": "user",
                    "content": news_text
                }
            ]
        
        response = self.client.chat.completions.create(
            messages=messages,
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
            model=self.deployment_name
        )
        try:
            self.ner_result = json.loads(response.choices[0].message.content.strip().strip('`').strip("json"))
        except Exception as e:
            print(f"error fetching NER result: {e}", response.choices[0].message.content)
    
    def classify_news(self, news_text):
        messages=[
            {
                    "role": "system",
                    "content":
                    f"""
                    You are an expert in dentifying adverse financial news and classifying them. I will be providing a set of adverse news categories and some news articles containing [title] and [snippet], and you will need to classify the articles into the categories.

                    ## Constraints
                    1. Return a parsable JSON object with the following fields: category, confidence_score (the score should show how confidence you are that the document belong to the category), justification(justify the category and score).
                    2. Only return the JSON output, nothing else. Do not include any other strings whatsoever, even it's for formatting.
                    3. If multiple news articles are provided, return multiple results nested in a list.
                    4. Keep the output format consistent for all the articles provided and preserve the order of the articles.

                    ## Categories
                    Money Laundering, Terrorist Financing, Sanctions Violations, Fraud, Tax Evasion, Bribery and Corruption, Insider Trading, Ponzi and Pyramid Schemes, Trade-Based Money Laundering, General Financial News, Non Financial News
                    """
                    
            },
            {
                    "role": "user",
                    "content": news_text
                }
        ]

        response = self.client.chat.completions.create(
            messages=messages,
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
            model=self.deployment_name
        )
        
        try:
            self.classification_result = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"error fetching classification result: {e}")
