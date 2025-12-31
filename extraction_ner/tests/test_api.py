"""
ClinicalTrials.gov API Testing Script

Purpose: Explore and validate the ClinicalTrials.gov API v2 structure
         to understand data availability for future enhancements.

Context: This script was used during development to:
         1. Understand API response structure
         2. Identify available fields for entity extraction
         3. Plan potential data expansion beyond the 91 annotated protocols

Status: Exploratory/development script - not part of production pipeline

Output: 
  - search_response.json (search API structure)
  - study_response.json (individual study structure)
"""

import requests
import json

def test_search_api():
    """Test the search API to find trials by condition"""
    print("=" * 70)
    print("TEST 1: SEARCH API")
    print("=" * 70)
    
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
            
            # Extract NCT IDs for further queries
            if 'studies' in data:
                print(f"\nFound 'studies' key with {len(data['studies'])} studies")
                if len(data['studies']) > 0:
                    first_study = data['studies'][0]
                    print(f"\nFirst study keys: {list(first_study.keys())}")
                    
                    # Navigate to NCT ID
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


def test_individual_study_api():
    """Test fetching a specific trial by NCT ID"""
    print("\n" + "=" * 70)
    print("TEST 2: INDIVIDUAL STUDY API")
    print("=" * 70)
    
    # Use a known pembrolizumab trial
    test_nct_id = "NCT02775435"
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
            
            # Navigate structure to find key sections
            if 'studies' in data and len(data['studies']) > 0:
                study = data['studies'][0]
                print(f"\nStudy keys: {list(study.keys())}")
                
                if 'protocolSection' in study:
                    protocol = study['protocolSection']
                    print(f"\nProtocol Section keys (first 10): {list(protocol.keys())[:10]}")
                    
                    # Check for description module
                    if 'descriptionModule' in protocol:
                        desc_module = protocol['descriptionModule']
                        print(f"\nDescription Module keys: {list(desc_module.keys())}")
                        
                        if 'briefSummary' in desc_module:
                            summary = desc_module['briefSummary']
                            print(f"\nBrief Summary (first 200 chars):")
                            print(summary[:200])
                    
                    # Check for eligibility criteria
                    if 'eligibilityModule' in protocol:
                        elig_module = protocol['eligibilityModule']
                        print(f"\nEligibility Module keys: {list(elig_module.keys())}")
                        
                        if 'eligibilityCriteria' in elig_module:
                            criteria = elig_module['eligibilityCriteria']
                            print(f"\nEligibility Criteria (first 200 chars):")
                            print(criteria[:200])
            else:
                print("Unexpected structure - check study_response.json")
        else:
            print(f"Error: {response.status_code}")
            print(response.text[:500])
    
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run API exploration tests"""
    test_search_api()
    test_individual_study_api()
    
    print("\n" + "=" * 70)
    print("API EXPLORATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("1. search_response.json - Search API structure")
    print("2. study_response.json - Individual study structure")
    print("\nUse these to understand available fields for entity extraction")


if __name__ == "__main__":
    main()
