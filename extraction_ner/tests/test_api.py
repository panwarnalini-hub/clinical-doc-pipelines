import requests
import json

def test_api_structure():
    
    # Test 1: Search API
    print("="*70)
    print("TEST 1: SEARCH API")
    print("="*70)
    
    search_url = "https://clinicaltrials.gov/api/v2/studies"
    search_params = {
        'query.cond': 'lung cancer',
        'query.term': 'Phase 3',
        'pageSize': 3,
        'format': 'json'
    }
    
    try:
        response = requests.get(search_url, params=search_params, timeout=15)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nTop-level keys: {list(data.keys())}")
            
            # Save full response for inspection
            with open('search_response.json', 'w') as f:
                json.dump(data, f, indent=2)
            print("Saved full response to: search_response.json")
            
            # Try to extract NCT IDs
            if 'studies' in data:
                print(f"\nFound 'studies' key with {len(data['studies'])} studies")
                if len(data['studies']) > 0:
                    first_study = data['studies'][0]
                    print(f"\nFirst study keys: {list(first_study.keys())}")
                    
                    # Try to get NCT ID
                    if 'protocolSection' in first_study:
                        protocol = first_study['protocolSection']
                        if 'identificationModule' in protocol:
                            nct_id = protocol['identificationModule'].get('nctId')
                            print(f"Extracted NCT ID: {nct_id}")
            else:
                print("No 'studies' key in response")
                print(f"Available keys: {list(data.keys())}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text[:500])
    
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Individual Study API
    print("\n" + "="*70)
    print("TEST 2: INDIVIDUAL STUDY API")
    print("="*70)
    
    # Use a known trial ID
    test_nct_id = "NCT02775435"  # Known pembrolizumab trial
    study_url = f"https://clinicaltrials.gov/api/v2/studies/{test_nct_id}"
    
    try:
        response = requests.get(study_url, timeout=15)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nTop-level keys: {list(data.keys())}")
            
            # Save full response
            with open('study_response.json', 'w') as f:
                json.dump(data, f, indent=2)
            print("Saved full response to: study_response.json")
            
            # Navigate structure
            if 'studies' in data and len(data['studies']) > 0:
                study = data['studies'][0]
                print(f"\nStudy keys: {list(study.keys())}")
                
                if 'protocolSection' in study:
                    protocol = study['protocolSection']
                    print(f"\nProtocol Section keys: {list(protocol.keys())[:10]}...")
                    
                    # Check for description
                    if 'descriptionModule' in protocol:
                        desc_module = protocol['descriptionModule']
                        print(f"\nDescription Module keys: {list(desc_module.keys())}")
                        
                        if 'briefSummary' in desc_module:
                            summary = desc_module['briefSummary']
                            print(f"\nBrief Summary (first 200 chars):")
                            print(summary[:200])
                    
                    # Check for eligibility
                    if 'eligibilityModule' in protocol:
                        elig_module = protocol['eligibilityModule']
                        print(f"\nEligibility Module keys: {list(elig_module.keys())}")
                        
                        if 'eligibilityCriteria' in elig_module:
                            criteria = elig_module['eligibilityCriteria']
                            print(f"\nEligibility Criteria (first 200 chars):")
                            print(criteria[:200])
            else:
                print("Unexpected structure")
        else:
            print(f"Error: {response.status_code}")
            print(response.text[:500])
    
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*70)
    print("DEBUG COMPLETE")
    print("="*70)
    print("\nPlease check:")
    print("1. search_response.json - to see search API structure")
    print("2. study_response.json - to see individual study structure")

if __name__ == "__main__":
    test_api_structure()
