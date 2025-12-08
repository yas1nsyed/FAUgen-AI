"""Script for scraper to fetch the websites"""
import requests
import re
from bs4 import BeautifulSoup


class WebsiteScraper:
    def __init__(self, header = None):
        self.header = header if header else {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cookie": "cookie_consent_accepted=true; other_cookie=other_value",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Fetch-User": "?1"
        }
        self.remove_tags = ["style", "script"]
        self.remove_class = []
        self.remove_ID = ["footer", "social", "headerwrapper"]

    def scrape_website(self, url: str) -> tuple[int, object]:
        """
        Scrape the main content from a given website link.
        Args:
            url (str): web url link for the website you want to scrape.
        """
        try:
            response = requests.get(url, timeout=10, headers=self.header, allow_redirects=True)
            if response.status_code != 200:
                return response.status_code, None
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            clean_format = self.clean_html_content(soup)
            title = soup.title.string if soup.title else "No title found"
            meta_description = soup.find('meta', property='og:description') if soup.find('meta', property='og:description') else "No meta description found"
            final_scraped_content = {
                "title": title,
                "meta_description": meta_description,
                "content": clean_format.get_text(separator=' ', strip=True)
            }
            return  response.status_code ,final_scraped_content

        except requests.RequestException as e:
            print(f"Error fetching URL: {str(e)}")
            return None, None
        except Exception as e:
            print(f"Error processing URL: {str(e)}")
            return None, None

    def clean_html_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """
        Clean the HTML content by removing unwanted tags and attributes.
        Args:
            soup (BeautifulSoup): Parsed HTML content.
        """

        # Remove script and style elements
        for script in soup(self.remove_tags):
            script.decompose()

        # Remove unwanted ID from HTML
        unwanted_ids = self.remove_ID
        for id_name in unwanted_ids:
            for element in soup.find_all(id=re.compile(id_name, re.I)):
                element.decompose()
        
        #Remove unwanted class from the HTML
        unwanted_class = self.remove_class
        for class_name in unwanted_class:
            for element in soup.find_all(class_=re.compile(class_name, re.I)):
                element.decompose()

        return soup