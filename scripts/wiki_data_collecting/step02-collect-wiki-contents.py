# pip install wikipedia-api tqdm mwparserfromhell requests

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import requests
import mwparserfromhell
from tqdm import tqdm
import re

class WikiContentExtractor:
    def __init__(self, output_dir='wiki-contents', max_workers=3, delay=0.5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.delay = delay
        self.lock = Lock()
        self.api_url = "https://ar.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikiContentExtractor/1.0 (Educational Project; Python/requests)'
        })
        
    def get_page_id_from_url(self, url):
        """Extract page title from Wikipedia URL."""
        # Extract the part after /wiki/
        match = re.search(r'/wiki/(.+)$', url)
        if match:
            return match.group(1)
        return None
    
    def fetch_page_wikitext(self, title):
        """Fetch the raw wikitext of a Wikipedia page."""
        from urllib.parse import unquote
        
        # Decode the title for the API request
        decoded_title = unquote(title).replace('_', ' ')
        
        params = {
            "action": "query",
            "format": "json",
            "titles": decoded_title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main"
        }
        
        try:
            response = self.session.get(self.api_url, params=params, timeout=10)
            
            # Check if response is valid
            if response.status_code != 200:
                print(f"HTTP Error {response.status_code}: {response.text[:200]}")
                return None
            
            # Try to parse JSON
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Response text: {response.text[:500]}")
                return None
            
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if page_id == '-1':
                    return None  # Page doesn't exist
                
                revisions = page_data.get('revisions', [])
                if revisions:
                    return revisions[0]['slots']['main']['*']
            
            return None
        except Exception as e:
            print(f"Error fetching wikitext for '{decoded_title}': {str(e)}")
            return None
    
    def wikitext_to_markdown(self, wikitext):
        """Convert wikitext to markdown, preserving lists and structure."""
        if not wikitext:
            return ""
        
        # Parse the wikitext
        wikicode = mwparserfromhell.parse(wikitext)
        
        # Extract plain text but keep some structure
        markdown = ""
        lines = str(wikicode).split('\n')
        in_list = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines, templates, and metadata
            if not line or line.startswith('{{') or line.startswith('[[Category:') or line.startswith('[[تصنيف:'):
                continue
            
            # Handle headings
            if line.startswith('=='):
                # Count the number of = signs
                level = 0
                for char in line:
                    if char == '=':
                        level += 1
                    else:
                        break
                level = level // 2
                
                # Extract heading text
                heading = line.strip('= ').strip()
                # Clean internal links from heading
                heading = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', heading)
                markdown += f"\n{'#' * (level + 1)} {heading}\n\n"
                in_list = False
                continue
            
            # Handle list items (bullets)
            if line.startswith('*'):
                # Remove the * and clean the text
                item = line[1:].strip()
                # Clean wikilinks: [[link|text]] -> text, [[link]] -> link
                item = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', item)
                # Clean bold/italic
                item = re.sub(r"'''([^']+)'''", r'**\1**', item)
                item = re.sub(r"''([^']+)''", r'*\1*', item)
                # Clean references
                item = re.sub(r'<ref[^>]*>.*?</ref>', '', item)
                item = re.sub(r'<ref[^>]*\/>', '', item)
                
                markdown += f"- {item}\n"
                in_list = True
                continue
            
            # Handle numbered lists
            if line.startswith('#'):
                item = line[1:].strip()
                item = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', item)
                item = re.sub(r"'''([^']+)'''", r'**\1**', item)
                item = re.sub(r"''([^']+)''", r'*\1*', item)
                item = re.sub(r'<ref[^>]*>.*?</ref>', '', item)
                item = re.sub(r'<ref[^>]*\/>', '', item)
                
                markdown += f"1. {item}\n"
                in_list = True
                continue
            
            # Handle regular paragraphs
            if not line.startswith('{') and not line.startswith('|') and not line.startswith('!'):
                # Clean the text
                text = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', line)
                text = re.sub(r"'''([^']+)'''", r'**\1**', text)
                text = re.sub(r"''([^']+)''", r'*\1*', text)
                text = re.sub(r'<ref[^>]*>.*?</ref>', '', text)
                text = re.sub(r'<ref[^>]*\/>', '', text)
                
                if text and not text.startswith('thumb|'):
                    if in_list:
                        markdown += "\n"
                        in_list = False
                    markdown += f"{text}\n\n"
        
        return markdown
    
    def is_already_extracted(self, title):
        """Check if content is already extracted."""
        safe_title = self.sanitize_filename(title)[:100]
        filename = f"{safe_title}.md"
        filepath = self.output_dir / filename
        return filepath.exists()
    
    def sanitize_filename(self, text):
        """Remove invalid filename characters."""
        return re.sub(r'[<>:"/\\|?*]', '_', text)
    
    def extract_content(self, article_data, force=False):
        """Extract article content and save as markdown."""
        title = article_data['title']
        
        # Decode title for display and filename
        from urllib.parse import unquote
        display_title = unquote(title).replace('_', ' ')
        
        # Check if already extracted
        if not force and self.is_already_extracted(display_title):
            return False
        
        # Rate limiting
        time.sleep(self.delay)
        
        try:
            # Fetch the raw wikitext using the original encoded title
            wikitext = self.fetch_page_wikitext(title)
            
            if not wikitext:
                return False
            
            # Build markdown content using display title
            markdown = f"# {display_title}\n\n"
            
            # Metadata section
            markdown += "## [METADATA]\n\n"
            markdown += f"- **URL:** {article_data['url']}\n"
            markdown += f"- **Category:** {article_data.get('category', 'N/A')}\n"
            
            if article_data.get('parent_categories'):
                markdown += f"- **Parent Categories:** {' > '.join(article_data['parent_categories'])}\n"
            
            markdown += "\n---\n\n"
            
            # Main content section
            markdown += "## [CONTENT]\n\n"
            
            # Convert wikitext to markdown
            content_markdown = self.wikitext_to_markdown(wikitext)
            markdown += content_markdown
            
            # Save to file using display title
            safe_title = self.sanitize_filename(display_title)[:100]  # Limit length
            filename = f"{safe_title}.md"
            filepath = self.output_dir / filename
            
            with self.lock:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(markdown)
            
            return True
            
        except Exception as e:
            print(f"Error extracting '{display_title}': {str(e)}")
            return False
    
    def process_articles(self, articles, force=False):
        """Process articles in parallel."""
        extracted_count = 0
        skipped_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.extract_content, article, force): article for article in articles}
            
            with tqdm(total=len(articles), desc="Extracting content", unit=" articles") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        extracted_count += 1
                    else:
                        skipped_count += 1
                    pbar.update(1)
                    pbar.set_postfix({'extracted': extracted_count, 'skipped': skipped_count})
        
        return extracted_count, skipped_count


if __name__ == "__main__":
    import argparse
    from urllib.parse import unquote
    
    parser = argparse.ArgumentParser(description='Extract Wikipedia articles content')
    parser.add_argument('--url', type=str, help='Single Wikipedia URL to extract')
    parser.add_argument('--force-extract', action='store_true', help='Force re-extraction and overwrite existing files')
    args = parser.parse_args()
    
    extractor = WikiContentExtractor(output_dir='wiki-contents', max_workers=3, delay=0.5)
    
    if args.url:
        # Extract single URL
        title = extractor.get_page_id_from_url(args.url)
        if title:
            # Keep title encoded for API, but decode for display
            display_title = unquote(title).replace('_', ' ')
            article = {'url': args.url, 'title': title, 'category': 'N/A', 'parent_categories': []}
            print(f"Extracting single article: {display_title}")
            success = extractor.extract_content(article, force=args.force_extract)
            if success:
                print(f"✓ Successfully extracted to wiki-contents/{extractor.sanitize_filename(display_title)[:100]}.md")
            else:
                print(f"✗ Failed to extract or already exists")
    else:
        # Process all articles from file
        input_file = 'articles-urls.jsonl'
        
        articles_dict = {}
        total_loaded = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                total_loaded += 1
                article = json.loads(line.strip())
                url = article['url']
                if url not in articles_dict:
                    articles_dict[url] = article
        
        articles = list(articles_dict.values())
        duplicates = total_loaded - len(articles)
        
        print(f"Loaded {total_loaded} entries from {input_file}")
        print(f"Found {len(articles)} unique articles (removed {duplicates} duplicates)")
        
        extracted, skipped = extractor.process_articles(articles, force=args.force_extract)
        
        print(f"\n{'='*60}")
        print(f"Extraction complete!")
        print(f"Extracted: {extracted} articles")
        print(f"Skipped: {skipped} articles")
        print(f"Output directory: {extractor.output_dir}")
        print(f"{'='*60}")
