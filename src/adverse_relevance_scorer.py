class AdverseRelevanceScorer:
    def __init__(self, entity_dict, classification_confidence):
        self.entity_dict = entity_dict
        self.classification_confidence = classification_confidence
        self.entity_type_score = 0
        self.comb_bonus = 0
        self.relevance_score = 0

    def compute_entity_type_score(self, entity_type_weights = None):
        """
        Compute entity related score based on different weights on entity types and raw mention count of entities.
        """
        entity_type_weights = entity_type_weights if  entity_type_weights else \
            {
                "COMPANY": 0.7,
                "PERSON": 0.9,
                "FINANCIAL_INSTITUTION": 0.9,
                "REGULATORY_BODY": 0.8,
                "PROTENTIAL_CRIME": 1, 
                "LEGAL_ACTION": 1, 
                "ENFORCEMENT_ACTION": 1,
                "LOCATION": 0.3,
                "SANCTION_ENTITY": 0.9,
                "SECTOR": 0.3,
                "REGULATION": 0.4
            }
        score = 0.0
        for ent_type, entities in self.entity_dict.items():
            if isinstance(entities, list):
                score += entity_type_weights[ent_type] * len(entities)
                    
        self.entity_type_score = min(round(score /len(self.entity_dict), 2), 1.0)  # cap to 1.0
        
    def compute_combination_bonus(self, max_bonus=0.2):
        """
        Calculate bonus score for high risk combinations: LEGAL_ACTION + PERSON, PROTENTIAL_CRIME + PERSON and ENFORCEMENT_ACTION + COMPANY, 
        (Capped at max_bonus)
        """
        flags = 0
        if len(self.entity_dict.get("LEGAL_ACTION", [])) > 0 and len(self.entity_dict.get("PERSON", [])) > 0:
            flags += 1
        if len(self.entity_dict.get("PROTENTIAL_CRIME", [])) > 0 and len(self.entity_dict.get("PERSON", [])) > 0:
            flags += 1
        if len(self.entity_dict.get("ENFORCEMENT_ACTION", [])) > 0 and len(self.entity_dict.get("COMPANY", [])) > 0:
            flags += 1
        self.comb_bonus = min(flags * 0.1, max_bonus)  # Add up to +0.2 bonus 
    
    def compute_relevant_score(self, components_weight=None):
        components_weight = components_weight if components_weight else \
        {
            "classification_confidence": 0.5,
            "entity_type": 0.5
        }
        self.compute_entity_type_score()
        self.compute_combination_bonus()
        self.relevance_score = round(
            self.classification_confidence * components_weight["classification_confidence"] + \
            self.entity_type_score * components_weight["entity_type"] + \
            self.comb_bonus,
            3
        )
            
            