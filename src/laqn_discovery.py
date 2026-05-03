import requests
import json

BASE_URL = "https://api.londonair.org.uk/api"

def get_sites():
    print("Fetching sites...")
    res = requests.get(f"{BASE_URL}/Information/AllSites/Json")
    res.raise_for_status()
    data = res.json()
    sites = data['Sites']['Site']
    
    # Filter for sites that are currently active and have multiple species
    for s in sites:
        code = s['@SiteCode']
        name = s['@SiteName']
        print(f"Code: {code} | Name: {name}")

if __name__ == "__main__":
    get_sites()
