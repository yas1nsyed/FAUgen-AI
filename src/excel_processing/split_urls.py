import pandas as pd
import requests
from urllib.parse import urlparse, urljoin

def split_urls_to_columns(excel_file, output_file='split_urls.xlsx'):
    """
    Split URLs into separate columns after each '/'
    """
    try:
        # Read Excel file
        df = pd.read_excel(excel_file)
        
        # Find the column containing URLs (assuming it's named 'URL')
        url_column = None
        for col in df.columns:
            if 'url' in col.lower():
                url_column = col
                break
        
        if url_column is None:
            # If no URL column found, use first column
            url_column = df.columns[0]
            print(f"No URL column found. Using first column: {url_column}")
        
        # Create new columns for URL parts
        max_parts = 0
        
        # First, find the maximum number of parts
        for url in df[url_column]:
            if pd.notna(url):
                parts = str(url).split('/')
                max_parts = max(max_parts, len(parts))
        
        # Create new columns
        new_columns = []
        for i in range(max_parts):
            col_name = f'Part_{i+1}'
            new_columns.append(col_name)
            df[col_name] = None
        
        # Split URLs and populate columns
        for idx, url in enumerate(df[url_column]):
            if pd.notna(url):
                url_str = str(url)
                parts = url_str.split('/')
                
                for i, part in enumerate(parts):
                    if i < max_parts:
                        df.at[idx, f'Part_{i+1}'] = part if part else ''
        
        # Save to new Excel file
        df.to_excel(output_file, index=False)
        print(f"Successfully split URLs and saved to {output_file}")
        print(f"Original file: {excel_file}")
        print(f"Created {max_parts} additional columns")
        
        return df
    
    except Exception as e:
        print(f"Error: {e}")
        return None

# Usage - replace with your actual Excel file path
split_urls_to_columns('sitemap_sites.xlsx', 'split_urls_output.xlsx')