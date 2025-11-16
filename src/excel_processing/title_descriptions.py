import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse
import re

def extract_metadata_from_urls(excel_file, output_file='urls_with_metadata.xlsx', delay=1):
    """
    Read URLs from Excel, extract title and description from each page
    """
    try:
        # Read Excel file
        df = pd.read_excel(excel_file)
        print(f"Loaded {len(df)} URLs from {excel_file}")
        
        # Find URL column
        url_column = None
        for col in df.columns:
            if 'url' in col.lower():
                url_column = col
                break
        
        if url_column is None:
            url_column = df.columns[0]
            print(f"No URL column found. Using first column: {url_column}")
        
        # Add new columns for metadata
        df['Page_Title'] = ''
        df['Meta_Description'] = ''
        df['Status_Code'] = ''
        df['Error'] = ''
        
        # Configure requests session
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Process each URL
        for idx, url in enumerate(df[url_column]):
            if pd.notna(url) and url != '':
                url_str = str(url).strip()
                print(f"Processing {idx+1}/{len(df)}: {url_str}")
                
                try:
                    # Make request
                    response = session.get(url_str, timeout=10)
                    df.at[idx, 'Status_Code'] = response.status_code
                    
                    if response.status_code == 200:
                        # Parse HTML
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Extract title
                        title = soup.find('title')
                        if title:
                            df.at[idx, 'Page_Title'] = title.get_text().strip()
                        
                        # Extract description - try multiple methods
                        description = ''
                        
                        # Method 1: og:description
                        og_desc = soup.find('meta', property='og:description')
                        if og_desc and og_desc.get('content'):
                            description = og_desc['content'].strip()
                        
                        # Method 2: name="description"
                        if not description:
                            meta_desc = soup.find('meta', attrs={'name': 'description'})
                            if meta_desc and meta_desc.get('content'):
                                description = meta_desc['content'].strip()
                        
                        # Method 3: twitter:description
                        if not description:
                            twitter_desc = soup.find('meta', attrs={'name': 'twitter:description'})
                            if twitter_desc and twitter_desc.get('content'):
                                description = twitter_desc['content'].strip()
                        
                        df.at[idx, 'Meta_Description'] = description
                        
                        print(f"  ✓ Title: {df.at[idx, 'Page_Title'][:50]}...")
                        print(f"  ✓ Description: {description[:50]}...")
                    
                    else:
                        df.at[idx, 'Error'] = f"HTTP {response.status_code}"
                        print(f"  ✗ HTTP Error: {response.status_code}")
                
                except requests.exceptions.RequestException as e:
                    error_msg = str(e)
                    df.at[idx, 'Error'] = error_msg
                    print(f"  ✗ Request Error: {error_msg}")
                
                except Exception as e:
                    error_msg = str(e)
                    df.at[idx, 'Error'] = error_msg
                    print(f"  ✗ Processing Error: {error_msg}")
                
                # Delay between requests to be respectful
                time.sleep(delay)
            
            else:
                df.at[idx, 'Error'] = 'Empty URL'
                print(f"Skipping empty URL at row {idx+1}")
        
        # Save results
        df.to_excel(output_file, index=False)
        print(f"\n✅ Successfully processed {len(df)} URLs")
        print(f"📁 Results saved to: {output_file}")
        
        # Print summary
        successful = len(df[df['Status_Code'] == 200])
        failed = len(df) - successful
        print(f"📊 Summary: {successful} successful, {failed} failed")
        
        return df
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def extract_detailed_metadata(excel_file, output_file='detailed_metadata.xlsx', delay=1):
    """
    Extract more detailed metadata including other meta tags
    """
    try:
        df = pd.read_excel(excel_file)
        print(f"Loaded {len(df)} URLs from {excel_file}")
        
        # Find URL column
        url_column = None
        for col in df.columns:
            if 'url' in col.lower():
                url_column = col
                break
        if url_column is None:
            url_column = df.columns[0]
        
        # Add comprehensive metadata columns
        df['Page_Title'] = ''
        df['Meta_Description'] = ''
        df['OG_Description'] = ''
        df['Twitter_Description'] = ''
        df['H1_Header'] = ''
        df['Canonical_URL'] = ''
        df['Status_Code'] = ''
        df['Content_Type'] = ''
        df['Error'] = ''
        
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        for idx, url in enumerate(df[url_column]):
            if pd.notna(url) and url != '':
                url_str = str(url).strip()
                print(f"Processing {idx+1}/{len(df)}: {url_str}")
                
                try:
                    response = session.get(url_str, timeout=10)
                    df.at[idx, 'Status_Code'] = response.status_code
                    df.at[idx, 'Content_Type'] = response.headers.get('content-type', '')
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Title
                        title = soup.find('title')
                        if title:
                            df.at[idx, 'Page_Title'] = title.get_text().strip()
                        
                        # Various description types
                        og_desc = soup.find('meta', property='og:description')
                        if og_desc:
                            df.at[idx, 'OG_Description'] = og_desc.get('content', '').strip()
                        
                        twitter_desc = soup.find('meta', attrs={'name': 'twitter:description'})
                        if twitter_desc:
                            df.at[idx, 'Twitter_Description'] = twitter_desc.get('content', '').strip()
                        
                        meta_desc = soup.find('meta', attrs={'name': 'description'})
                        if meta_desc:
                            df.at[idx, 'Meta_Description'] = meta_desc.get('content', '').strip()
                        
                        # H1 header
                        h1 = soup.find('h1')
                        if h1:
                            df.at[idx, 'H1_Header'] = h1.get_text().strip()
                        
                        # Canonical URL
                        canonical = soup.find('link', rel='canonical')
                        if canonical:
                            df.at[idx, 'Canonical_URL'] = canonical.get('href', '')
                        
                        print(f"  ✓ Title: {df.at[idx, 'Page_Title'][:30]}...")
                    
                    else:
                        df.at[idx, 'Error'] = f"HTTP {response.status_code}"
                
                except Exception as e:
                    df.at[idx, 'Error'] = str(e)
                
                time.sleep(delay)
        
        # Save results
        df.to_excel(output_file, index=False)
        print(f"\n✅ Detailed metadata extraction complete: {output_file}")
        
        return df
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Usage examples
if __name__ == "__main__":
    # Replace with your actual Excel file name
    excel_filename = "sitemap_sites.xlsx"  # Change this to your file name
    
    # Basic metadata extraction
    print("=== Basic Metadata Extraction ===")
    result = extract_metadata_from_urls(
        excel_file=excel_filename,
        output_file='urls_with_basic_metadata.xlsx',
        delay=0.5  # 0.5 second delay between requests
    )
    
    # Detailed metadata extraction (uncomment if needed)
    # print("\n=== Detailed Metadata Extraction ===")
    # detailed_result = extract_detailed_metadata(
    #     excel_file=excel_filename,
    #     output_file='urls_with_detailed_metadata.xlsx',
    #     delay=0.5
    # )