# pip install requests beautifulsoup4 tqdm mwparserfromhell openai json-repair

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import json
import time
import os
from tqdm import tqdm
import mwparserfromhell
from openai import OpenAI
import json_repair
from typing import Dict, List, Set
from dotenv import dotenv_values

config = dotenv_values("../.env")

class EgyptianFiguresCrawler:
    def __init__(self, output_file='./data/egyptian_figures_data.jsonl', max_workers=5, delay=0.1):
        self.output_file = output_file
        self.max_workers = max_workers
        self.delay = delay
        self.visited_categories = set()
        self.visited_figures = set()
        self.lock = Lock()
        self.figure_count = 0
        self.duplicates_skipped = 0
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        
        # OpenAI client for category classification
        self.openai_client = OpenAI(
            api_key=config.get("OPENAI_API_KEY")
        )
        
        # Load existing figures if resuming
        self._load_existing_figures()
        
    def _normalize_url(self, url):
        """Normalize URL to handle variations."""
        parsed = urlparse(url)
        path = unquote(parsed.path)
        path = path.rstrip('/')
        normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
        return normalized
    
    def _load_existing_figures(self):
        """Load previously collected figures to enable resume."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            normalized = self._normalize_url(data['url'])
                            self.visited_figures.add(normalized)
                            self.figure_count += 1
                print(f"Loaded {self.figure_count} existing figures for resume")
            except Exception as e:
                print(f"Warning: Could not load existing figures: {e}")
        
    def fetch_page(self, url):
        """Fetch a single page with rate limiting."""
        time.sleep(self.delay)
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def get_categories_in_wikipedia_category_page(self, content, base_url):
        """Extract subcategories from Wikipedia category page."""
        if not content:
            return []
            
        soup = BeautifulSoup(content, 'html.parser')
        categories = []
        
        # Find category groups
        category_groups = soup.find_all('div', class_='mw-category-group')
        for group in category_groups:
            links = group.find_all('a', href=True)
            for link in links:
                href = link['href']
                if href and '/wiki/Category' in href:
                    categories.append({
                        'name': link.get_text().strip(),
                        'link': urljoin(base_url, href)
                    })
        
        return categories
    
    def get_figures_in_category_page(self, content, base_url, category_name):
        """Extract figure pages (non-category links) from Wikipedia category page."""
        if not content:
            return []
            
        soup = BeautifulSoup(content, 'html.parser')
        figures = []
        
        # Find category groups
        category_groups = soup.find_all('div', class_='mw-category-group')
        for group in category_groups:
            links = group.find_all('a', href=True)
            for link in links:
                href = link['href']
                if href and '/wiki/Category' not in href and href.startswith('/wiki/'):
                    figures.append({
                        'name': link.get_text().strip(),
                        'link': urljoin(base_url, href),
                        'category': category_name
                    })
        
        return figures
    
    def classify_categories_with_openai(self, categories):
        """Classify categories as Egypt-related or not using OpenAI."""
        if not categories:
            return [], []
            
        egypt_related = []
        non_egypt_related = []
        
        batch_size = 40
        
        for i in range(0, len(categories), batch_size):
            batch = categories[i:i + batch_size]
            category_names = [cat['name'] for cat in batch]
            
            prompt = ("From the following list of Wikipedia categories, identify which ones are related to Egypt and which ones are not. "
                     "A category is considered Egypt-related if it pertains to Egypt's people, culture, history, geography, or significant aspects closely tied to Egypt. "
                     "Categories that are more general or pertain to other regions, cultures, or topics not specifically linked to Egypt should be classified as non-Egypt-related. "
                     f"Return a JSON object with two keys: 'egypt_related' and 'non_egypt_related', each containing a list of category names.\n\nCategories:\n{category_names}\n\nJSON:")
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that classifies Wikipedia categories."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                    temperature=0,
                )
                
                reply_content = response.choices[0].message.content
                classified_json = json_repair.loads(reply_content)
                
                egypt_related_names = classified_json.get("egypt_related", [])
                non_egypt_related_names = classified_json.get("non_egypt_related", [])
                
                for cat in batch:
                    if cat["name"] in egypt_related_names:
                        cat["is_egypt_related"] = True
                        egypt_related.append(cat)
                    elif cat["name"] in non_egypt_related_names:
                        cat["is_egypt_related"] = False
                        non_egypt_related.append(cat)
                    else:
                        cat["is_egypt_related"] = None
                        
            except Exception as e:
                print(f"Error classifying batch starting at index {i}: {e}")
                # On error, assume all categories in batch are Egypt-related to be safe
                for cat in batch:
                    cat["is_egypt_related"] = True
                    egypt_related.append(cat)
        
        return egypt_related, non_egypt_related
    
    def fetch_wikipedia_data(self, figure):
        """Fetch and parse Wikipedia data for a single figure."""
        try:
            # Extract page title from URL
            page_title = figure['link'].split('/wiki/')[-1]
            
            # API request
            api_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=revisions|pageprops&rvprop=content&format=json&titles={page_title}"
            response = requests.get(api_url, headers=self.headers, timeout=10)
            data = response.json()
            
            # Extract page content
            pages = data.get('query', {}).get('pages', {})
            if not pages:
                return None
                
            page_id = list(pages.keys())[0]
            page_data = pages[page_id]
            
            if 'revisions' not in page_data:
                return None
            
            wikitext = page_data['revisions'][0]['*']
            wikicode = mwparserfromhell.parse(wikitext)
            
            # Extract infobox
            infobox_data = {}
            templates = wikicode.filter_templates()
            for template in templates:
                if 'infobox' in template.name.lower().strip():
                    for param in template.params:
                        key = str(param.name).strip()
                        value = str(param.value).strip()
                        # Clean the value
                        value_code = mwparserfromhell.parse(value)
                        infobox_data[key] = value_code.strip_code()
            
            # Extract plain text (remove templates, links, etc.)
            plain_text = wikicode.strip_code()
            
            # Get first paragraph as summary (before first section)
            lines = plain_text.split('\n')
            summary = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('='):
                    summary.append(line)
                elif line.startswith('=='):
                    break
            summary_text = ' '.join(summary[:3])  # First 3 paragraphs
            
            return {
                'name': figure['name'],
                'url': figure['link'],
                'category': figure['category'],
                'title': page_data.get('title', ''),
                'summary': summary_text,
                'infobox': infobox_data
            }
            
        except Exception as e:
            print(f"Error processing {figure['name']}: {e}")
            return None
    
    def save_figure(self, figure_data):
        """Save figure data to jsonl file."""
        normalized_url = self._normalize_url(figure_data['url'])
        
        with self.lock:
            # Check if already collected
            if normalized_url in self.visited_figures:
                self.duplicates_skipped += 1
                return False
            
            # Mark as visited
            self.visited_figures.add(normalized_url)
            
            # Save to file
            with open(self.output_file, 'a', encoding='utf-8') as f:
                json.dump(figure_data, f, ensure_ascii=False)
                f.write('\n')
            self.figure_count += 1
            return True
    
    def crawl_categories_recursively(self, start_url, max_depth=3):
        """Crawl categories recursively to collect all subcategories."""
        print("Phase 1: Collecting categories...")
        
        all_categories = []
        to_visit = [(start_url, 0)]  # (url, depth)
        
        pbar = tqdm(desc="Collecting categories", unit=" cats")
        
        while to_visit:
            url, depth = to_visit.pop(0)
            
            normalized = self._normalize_url(url)
            if normalized in self.visited_categories or depth > max_depth:
                continue
                
            self.visited_categories.add(normalized)
            
            content = self.fetch_page(url)
            if content:
                categories = self.get_categories_in_wikipedia_category_page(content, url)
                all_categories.extend(categories)
                
                # Add subcategories to visit queue
                for cat in categories:
                    to_visit.append((cat['link'], depth + 1))
                
                pbar.update(1)
                pbar.set_postfix({'total_cats': len(all_categories), 'queue': len(to_visit)})
        
        pbar.close()
        
        # Remove duplicates
        unique_categories = []
        seen_links = set()
        for cat in all_categories:
            if cat['link'] not in seen_links:
                unique_categories.append(cat)
                seen_links.add(cat['link'])
        
        print(f"Collected {len(unique_categories)} unique categories")
        return unique_categories
    
    def filter_egypt_related_categories(self, categories):
        """Filter categories to only Egypt-related ones using OpenAI."""
        print("Phase 2: Classifying categories...")
        
        egypt_related, non_egypt_related = self.classify_categories_with_openai(categories)
        
        print(f"Egypt-related categories: {len(egypt_related)}")
        print(f"Non-Egypt-related categories: {len(non_egypt_related)}")
        
        return egypt_related
    
    def collect_figures_from_categories(self, categories):
        """Collect all figure pages from Egypt-related categories."""
        print("Phase 3: Collecting figures from categories...")
        
        all_figures = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for category in categories:
                future = executor.submit(self.fetch_page, category['link'])
                futures[future] = category
            
            pbar = tqdm(total=len(categories), desc="Collecting figures")
            
            for future in as_completed(futures):
                category = futures[future]
                content = future.result()
                
                if content:
                    figures = self.get_figures_in_category_page(content, category['link'], category['name'])
                    all_figures.extend(figures)
                
                pbar.update(1)
                pbar.set_postfix({'total_figures': len(all_figures)})
            
            pbar.close()
        
        # Remove duplicates
        unique_figures = []
        seen_links = set()
        for figure in all_figures:
            if figure['link'] not in seen_links:
                unique_figures.append(figure)
                seen_links.add(figure['link'])
        
        print(f"Collected {len(unique_figures)} unique figures")
        return unique_figures
    
    def process_figures_content(self, figures):
        """Process figures to extract their Wikipedia content."""
        print("Phase 4: Processing figure content...")
        
        batch_size = 10
        batches = [figures[i:i+batch_size] for i in range(0, len(figures), batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            pbar = tqdm(total=len(figures), desc="Processing figures")
            
            for batch in batches:
                futures = {executor.submit(self.fetch_wikipedia_data, figure): figure for figure in batch}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        self.save_figure(result)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'saved': self.figure_count,
                        'duplicates': self.duplicates_skipped
                    })
            
            pbar.close()
    
    def crawl(self, start_url, max_depth=3):
        """Main crawling method."""
        print(f"Starting Egyptian figures crawl from: {start_url}")
        print(f"Max depth: {max_depth}")
        print(f"Output file: {self.output_file}")
        print("=" * 60)
        
        # Phase 1: Collect all categories recursively
        all_categories = self.crawl_categories_recursively(start_url, max_depth)
        
        # Phase 2: Filter Egypt-related categories using OpenAI
        egypt_related_categories = self.filter_egypt_related_categories(all_categories)
        
        # Phase 3: Collect figures from Egypt-related categories
        all_figures = self.collect_figures_from_categories(egypt_related_categories)
        
        # Phase 4: Process figure content and save to file
        self.process_figures_content(all_figures)
        
        print("=" * 60)
        print(f"Crawl complete! Processed {self.figure_count} unique figures")
        print(f"Duplicates skipped: {self.duplicates_skipped}")
        print(f"Results saved to: {self.output_file}")
        print("=" * 60)


if __name__ == "__main__":
    start_url = "https://en.wikipedia.org/wiki/Category:Egyptian_people"
    
    crawler = EgyptianFiguresCrawler(
        output_file='./data/egyptian_figures_data.jsonl',
        max_workers=5,
        delay=0.1
    )
    
    crawler.crawl(start_url, max_depth=3)