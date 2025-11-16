import pandas as pd
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

def sitemap_to_excel(sitemap_url, output_file='sitemap_sites.xlsx'):
    """
    Extract all URLs from sitemap and save to Excel
    """
    try:
        # Fetch sitemap
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        # Namespace handling (common in sitemaps)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Extract URLs
        urls = []
        for url in root.findall('.//ns:url/ns:loc', namespace):
            urls.append(url.text)
        
        # Alternative namespace approach if above doesn't work
        if not urls:
            urls = [url.text for url in root.findall('.//loc')]
        
        # Create DataFrame
        df = pd.DataFrame(urls, columns=['URL'])
        
        # Save to Excel
        df.to_excel(output_file, index=False)
        print(f"Successfully saved {len(urls)} URLs to {output_file}")
        
        return df
    
    except Exception as e:
        print(f"Error: {e}")
        return None

# Usage
sitemap_url = "https://www.fau.de/sitemap.xml"  # Replace with your sitemap URL
sitemap_to_excel(sitemap_url)