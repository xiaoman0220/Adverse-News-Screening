import os
import json
import datetime
import requests

class NewsCollector:
    """
    Class to search for news based on query and time range
    """
    def __init__(self, api_key=None):
        self.result = None

        # Set news search APi configurations
        self.api_key = api_key if api_key else os.environ['NEWS_SEARCH_KEY']
        self.__time_range_map = {
            "Past month": "m",
            "Past week": "w",
            "Past year": "y"
        }

    def search(self, 
               query, 
               return_num=100, 
               time_range="Past month"):
        try:
            url = "https://google.serper.dev/news"
            params = {
                "q": f"related:business {query}",
                "num": return_num,
                }
            if time_range != "Any time":
                params["tbs"] = f"qdr:{self.__time_range_map[time_range]}"
            payload = json.dumps(params)
            headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
            }
            response = requests.request("POST", url, headers=headers, data=payload)
            self.result = response.json()
            

        except Exception as e:
            print(f"Error: {e}")




