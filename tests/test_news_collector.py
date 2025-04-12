import unittest
import sys
sys.path.append('../')

from src.news_collector import NewsCollector

class TestNewsCollector(unittest.TestCase):

    def test_search(self):
        collector = NewsCollector()
        collector.search('Sam Bankman-Fried')
        self.assertIsInstance(collector.result, list, 'The search result is wrong.')

if __name__ == '__main__':
    unittest.main()