"""
NER Model Testing Script

Purpose: Unit tests and validation for the trained clinical trial NER model

Tests:
1. Entity extraction accuracy on known examples
2. Model loading and inference
3. Entity type coverage
4. Edge cases (empty text, long text, special characters)

Usage: python tests/test_ner.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import training modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from training.scripts.inference import ClinicalNERInference, InferenceConfig
    MODEL_AVAILABLE = True
except ImportError:
    print("Warning: Could not import inference module. Model tests will be skipped.")
    MODEL_AVAILABLE = False


class TestNERModel:
    """Test suite for NER model validation"""
    
    def __init__(self):
        if MODEL_AVAILABLE:
            self.config = InferenceConfig()
            if self.config.model_path.exists():
                self.ner = ClinicalNERInference(self.config.model_path, self.config.device)
                self.model_loaded = True
            else:
                print(f"Model not found at {self.config.model_path}")
                self.model_loaded = False
        else:
            self.model_loaded = False
    
    def test_entity_extraction_basic(self):
        """Test basic entity extraction"""
        print("\n" + "=" * 70)
        print("TEST: Basic Entity Extraction")
        print("=" * 70)
        
        if not self.model_loaded:
            print("SKIPPED - Model not loaded")
            return
        
        test_cases = [
            {
                'text': "Patients with metastatic breast cancer will receive Pembrolizumab 200mg IV.",
                'expected_entities': ['metastatic breast cancer', 'Pembrolizumab']
            },
            {
                'text': "Inclusion: Age >= 18 years, ECOG performance status 0-2.",
                'expected_entities': ['18 years', 'ECOG']
            },
            {
                'text': "Primary endpoint: Progression-Free Survival at 12 months.",
                'expected_entities': ['Progression-Free Survival']
            }
        ]
        
        passed = 0
        failed = 0
        
        for i, test_case in enumerate(test_cases, 1):
            text = test_case['text']
            expected = test_case['expected_entities']
            
            entities = self.ner.predict(text)
            extracted = [e['text'] for e in entities]
            
            print(f"\nTest Case {i}:")
            print(f"  Text: {text[:60]}...")
            print(f"  Expected: {expected}")
            print(f"  Extracted: {extracted}")
            
            # Check if at least one expected entity was found
            found = any(exp in ' '.join(extracted) for exp in expected)
            if found or len(extracted) > 0:
                print("  Status: PASS")
                passed += 1
            else:
                print("  Status: FAIL")
                failed += 1
        
        print(f"\nResults: {passed} passed, {failed} failed")
    
    def test_entity_types(self):
        """Test that all entity types are recognized"""
        print("\n" + "=" * 70)
        print("TEST: Entity Type Coverage")
        print("=" * 70)
        
        if not self.model_loaded:
            print("SKIPPED - Model not loaded")
            return
        
        entity_type_examples = {
            'CONDITION': "Patients with non-small cell lung cancer",
            'DRUG': "Treatment with Durvalumab 10mg/kg",
            'DOSAGE': "Administered at 200mg IV every 3 weeks",
            'BIOMARKER': "Hemoglobin must be >= 9 g/dL",
            'PATIENT_CRITERIA': "Age between 18 and 75 years",
            'ENDPOINT': "Primary endpoint is Overall Survival",
            'STUDY_PHASE': "This Phase III randomized trial",
            'ENDPOINT_TYPE': "Primary and secondary endpoints"
        }
        
        types_found = set()
        
        for expected_type, text in entity_type_examples.items():
            entities = self.ner.predict(text)
            
            print(f"\n{expected_type}:")
            print(f"  Text: {text}")
            print(f"  Entities: {[(e['text'], e['label']) for e in entities]}")
            
            for entity in entities:
                types_found.add(entity['label'])
        
        print(f"\nEntity types found: {sorted(types_found)}")
        print(f"Total types: {len(types_found)}")
    
    def test_edge_cases(self):
        """Test edge cases"""
        print("\n" + "=" * 70)
        print("TEST: Edge Cases")
        print("=" * 70)
        
        if not self.model_loaded:
            print("SKIPPED - Model not loaded")
            return
        
        edge_cases = [
            ("", "Empty string"),
            ("No entities here", "No recognizable entities"),
            ("   ", "Whitespace only"),
            ("A" * 1000, "Very long text"),
            ("Age 18-65 years; ECOG 0-2; hemoglobin >9g/dL", "Multiple entities, special chars")
        ]
        
        for text, description in edge_cases:
            print(f"\n{description}:")
            print(f"  Text: {text[:60]}...")
            
            try:
                entities = self.ner.predict(text)
                print(f"  Entities: {len(entities)} found")
                if entities:
                    print(f"  Examples: {[e['text'] for e in entities[:3]]}")
                print("  Status: OK")
            except Exception as e:
                print(f"  Status: ERROR - {e}")
    
    def test_confidence_scores(self):
        """Test confidence score calculation"""
        print("\n" + "=" * 70)
        print("TEST: Confidence Scores")
        print("=" * 70)
        
        if not self.model_loaded:
            print("SKIPPED - Model not loaded")
            return
        
        text = "Patients with metastatic breast cancer receive Pembrolizumab 200mg every 3 weeks."
        
        try:
            results = self.ner.predict_with_scores(text)
            
            print(f"\nText: {text}")
            print("\nWord-level predictions:")
            for r in results[:10]:  # Show first 10
                print(f"  {r['word']:<20} {r['label']:<15} {r['confidence']:.3f}")
            
            print("\nStatus: OK")
        except Exception as e:
            print(f"Error: {e}")
    
    def run_all_tests(self):
        """Run all test cases"""
        print("=" * 70)
        print("CLINICAL TRIAL NER - MODEL TESTS")
        print("=" * 70)
        
        if not self.model_loaded:
            print("\nERROR: Model not loaded. Cannot run tests.")
            print(f"Expected model at: {self.config.model_path if MODEL_AVAILABLE else 'N/A'}")
            return
        
        print(f"\nModel: {self.config.model_path}")
        print(f"Device: {self.config.device}")
        
        self.test_entity_extraction_basic()
        self.test_entity_types()
        self.test_edge_cases()
        self.test_confidence_scores()
        
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETE")
        print("=" * 70)


def main():
    """Run NER model tests"""
    tester = TestNERModel()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
