import requests
import json

BASE_URL = "https://api.openaq.org/v3"

def get_parameters():
    print("Fetching parameters...")
    res = requests.get(f"{BASE_URL}/parameters")
    res.raise_for_status()
    data = res.json()
    params = {p['name']: p['id'] for p in data['results']}
    print(json.dumps(params, indent=2))
    return params

def find_locations(param_ids):
    print(f"Finding locations with parameters: {param_ids}")
    # Filter for locations having at least PM25, NO2, O3
    # Note: the API might require a comma separated list or different format
    p_str = ",".join(map(str, param_ids))
    res = requests.get(f"{BASE_URL}/locations?parameters_id={p_str}&limit=20")
    res.raise_for_status()
    data = res.json()
    for loc in data['results']:
        print(f"ID: {loc['id']} | Name: {loc['name']} | Sensors: {[s['parameter']['name'] for s in loc['sensors']]}")

if __name__ == "__main__":
    params = get_parameters()
    # IDs for pm25, no2, o3, co might vary. Let's find them from output first.
    # Usually: pm25=2, no2=1, o3=3, co=5 (checking...)
    target_names = ['pm25', 'no2', 'o3', 'co']
    target_ids = [params[name] for name in target_names if name in params]
    find_locations(target_ids)
