import unittest
import sys
sys.path.append('../')

from src.adverse_relevance_scorer import AdverseRelevanceScorer

class TestAdverseRelevanceScorer(unittest.TestCase):

    def test_scoring(self):
        entity_dict = \
            {  
                "COMPANY": [],  
                "PERSON": [  
                    {  
                        "entity_name": "Joe Anderson",  
                        "variations": ["Joe Anderson", "Former Liverpool mayor Joe Anderson"]  
                    },  
                    {  
                        "entity_name": "Derek Hatton",  
                        "variations": ["Derek Hatton", "city politician Derek Hatton"]  
                    }  
                ],  
                "FINANCIAL_INSTITUTION": [],  
                "REGULATORY_BODY": [],  
                "PROTENTIAL_CRIME": [  
                    {  
                        "entity_name": "bribery",  
                        "variations": ["bribery", "charges of bribery"]  
                    },  
                    {  
                        "entity_name": "misconduct",  
                        "variations": ["misconduct"]  
                    }  
                ],  
                "LEGAL_ACTION": [  
                    {  
                        "entity_name": "court appearance",  
                        "variations": ["court appearance", "appeared in court"]  
                    }  
                ],  
                "ENFORCEMENT_ACTION": [],  
                "LOCATION": [  
                    {  
                        "entity_name": "Liverpool",  
                        "variations": ["Liverpool"]  
                    }  
                ],  
                "SANCTION_ENTITY": [],  
                "SECTOR": [],  
                "REGULATION": []  
            }   
        classification_confidence = 0.95
        scorer = AdverseRelevanceScorer(entity_dict, classification_confidence)
        scorer.compute_relevant_score()
        self.assertEqual(scorer.relevance_score, 0.905, "The score calculation is incorrect.")

if __name__ == '__main__':
    unittest.main()