import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_global_analysis():
    print("Running Global 2025 Air Quality Analysis...")
    df = pd.read_csv("data/global_air_quality_2025.csv")
    
    # Create figures directory
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # 1. Country Comparison: Top 10 Worst vs Top 10 Best
    countries = df[df['Type'] == 'Country'].copy()
    countries = countries.sort_values('PM2.5_Average', ascending=False)
    top_worst = countries.head(10)
    top_best = countries.tail(10)
    
    comparison_df = pd.concat([top_worst, top_best])
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=comparison_df, x='PM2.5_Average', y='Name', palette='RdYlGn_r')
    plt.axvline(5, color='red', linestyle='--', label='WHO Guideline (5 µg/m³)')
    plt.title('Global Air Quality 2025: Most Polluted vs Cleanest Countries', fontsize=15)
    plt.xlabel('PM2.5 Average (µg/m³)', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/global_country_comparison_2025.png')
    print("Saved global_country_comparison_2025.png")

    # 2. City Comparison: Top 10 Worst vs Top 10 Best
    cities = df[df['Type'] == 'City'].copy()
    # Note: the input combined 'Rank' for worst/best for cities, let's just sort by value
    cities = cities.sort_values('PM2.5_Average', ascending=False)
    top_cities_worst = cities.head(10)
    top_cities_best = cities.tail(10)
    
    city_comparison_df = pd.concat([top_cities_worst, top_cities_best])
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=city_comparison_df, x='PM2.5_Average', y='Name', palette='OrRd_r')
    plt.axvline(5, color='blue', linestyle='--', label='WHO Guideline')
    plt.title('Global Air Quality 2025: Most Polluted vs Cleanest Cities', fontsize=15)
    plt.xlabel('PM2.5 Average (µg/m³)', fontsize=12)
    plt.ylabel('City', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/global_city_comparison_2025.png')
    print("Saved global_city_comparison_2025.png")

    # 3. Summary Stats
    print("\n--- 2025 Air Quality Highlights ---")
    print(f"Worst Country: {countries.iloc[0]['Name']} ({countries.iloc[0]['PM2.5_Average']} µg/m³)")
    print(f"Cleanest Country: {countries.iloc[-1]['Name']} ({countries.iloc[-1]['PM2.5_Average']} µg/m³)")
    print(f"Worst City: {cities.iloc[0]['Name']} ({cities.iloc[0]['PM2.5_Average']} µg/m³)")
    print(f"Cleanest City: {cities.iloc[-1]['Name']} ({cities.iloc[-1]['PM2.5_Average']} µg/m³)")

if __name__ == "__main__":
    run_global_analysis()
