import unittest
import sys
sys.path.append('../')

from src.llm_news_analyzer import LLMNewsAnalyzer

class TestLLMNewsAnalyzer(unittest.TestCase):

    def test_ner(self):
        analyzer = LLMNewsAnalyzer()
        news_text = """
        [title]Elizabeth Holmes vs. Sam Bankman-Fried: A Tale of Two CEO Scandals [snippet]Explore the rise and fall of Elizabeth Holmes and Sam Bankman-Fried—two CEOs who built billion-dollar companies before facing fraud charges.
        """

        analyzer.extract_entities(news_text)
        self.assertIsInstance(analyzer.ner_result, list, "The NER result is invalid.")
    
    def test_classification(self):
        analyzer = LLMNewsAnalyzer()
        news_text = """
        [title]Elizabeth Holmes vs. Sam Bankman-Fried: A Tale of Two CEO Scandals [snippet]Explore the rise and fall of Elizabeth Holmes and Sam Bankman-Fried—two CEOs who built billion-dollar companies before facing fraud charges.
        """

        analyzer.classify_news(news_text)
        self.assertEqual(analyzer.classification_result[0]["category"], "Fraud", "The classification is inconsistent.")
        self.assertEqual(analyzer.classification_result[0]["confidence_score"], 0.95, "The confidence score is inconsistent.")

if __name__ == '__main__':
    unittest.main()