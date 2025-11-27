"""Script for extracting data and save inside excel file"""
import os
import pandas as pd
from src.tools.scraper import WebsiteScraper

def website_to_excel():
    """ Function to get extract the data and save inside excel file. """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(base_dir, "./aces_metadata.xlsx")
    excel_df = pd.read_excel(excel_path)
    
    if "content" not in excel_df.columns:
        excel_df["content"] = ""
    try:
        for index, row in excel_df.iterrows():
            print(f"Scraping URL: {row["URL"]}")
            scraper = WebsiteScraper()
            status_code, content = scraper.scrape_website(row["URL"])
            if status_code != 200:
                print(f"Failed to scrape {row["URL"]} with status code {status_code}")
                continue
            excel_df.at[index, "content"] = content
        
        excel_df.to_excel(os.path.join(base_dir, "./aces_metadata_output.xlsx"), index=False)
    except Exception as e:
        print(f"Error: {e}")