# pip install requests beautifulsoup4 tqdm

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote, urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import json
import time
import os
from tqdm import tqdm

class WikiCrawler:
    def __init__(self, output_file='./data/articles-urls.jsonl', max_workers=3, delay=0.5):
        self.output_file = output_file
        self.max_workers = max_workers
        self.delay = delay
        self.visited = set()
        self.visited_articles = set()
        self.lock = Lock()
        self.article_count = 0
        self.duplicates_skipped = 0
        self.category_path = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Load existing articles if resuming
        self._load_existing_articles()
        
    def _normalize_url(self, url):
        """Normalize URL to handle variations."""
        # Parse URL
        parsed = urlparse(url)
        
        # Get path and decode it
        path = unquote(parsed.path)
        
        # Remove trailing slash
        path = path.rstrip('/')
        
        # Reconstruct without query params for deduplication
        normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
        
        return normalized
    
    def _load_existing_articles(self):
        """Load previously collected articles to enable resume."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            normalized = self._normalize_url(data['url'])
                            self.visited_articles.add(normalized)
                            self.article_count += 1
                print(f"Loaded {self.article_count} existing articles for resume")
            except Exception as e:
                print(f"Warning: Could not load existing articles: {e}")
        
    def fetch_page(self, url):
        """Fetch a single page with rate limiting."""
        time.sleep(self.delay)
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.content
        except Exception as e:
            return None
    
    def parse_category(self, content, base_url):
        """Extract subcategories and articles from category page."""
        soup = BeautifulSoup(content, 'html.parser')
        subcats = []
        articles = []
        
        # Validate this is actually a category page by checking URL
        if not any(marker in base_url for marker in ['تصنيف:', '%D8%AA%D8%B5%D9%86%D9%8A%D9%81:', 'Category:']):
            return subcats, articles, '', []
        
        # Get category name from the page title
        title_elem = soup.find('h1', class_='firstHeading')
        category_name = title_elem.get_text() if title_elem else ''
        category_name = category_name.replace('تصنيف:', '').strip()
        
        # Find subcategories
        subcats_div = soup.find('div', {'id': 'mw-subcategories'})
        if subcats_div:
            for link in subcats_div.find_all('a', href=True):
                href = link['href']
                if href.startswith('/wiki/'):
                    # Category pages contain تصنيف: in the URL
                    if '%D8%AA%D8%B5%D9%86%D9%8A%D9%81:' in href or 'تصنيف:' in href:
                        subcat_name = link.get_text().strip()
                        subcats.append({
                            'url': urljoin(base_url, href),
                            'name': subcat_name
                        })
        
        # Find articles
        pages_div = soup.find('div', {'id': 'mw-pages'})
        if pages_div:
            for link in pages_div.find_all('a', href=True):
                href = link['href']
                if href.startswith('/wiki/'):
                    # Exclude special pages, namespaces, and category pages
                    # These markers indicate non-article pages
                    excluded_markers = [
                        'تصنيف:', '%D8%AA%D8%B5%D9%86%D9%8A%D9%81:',  # Category
                        'Special:', 'خاص:',  # Special pages
                        'User:', 'مستخدم:',  # User pages
                        'File:', 'ملف:', 'صورة:',  # Files/images
                        'Template:', 'قالب:',  # Templates
                        'Talk:', 'نقاش:',  # Talk pages
                        'Wikipedia:', 'ويكيبيديا:',  # Wikipedia namespace
                        'Help:', 'مساعدة:',  # Help pages
                        'Portal:', 'بوابة:',  # Portals
                    ]
                    
                    # Check if URL contains any excluded markers
                    if not any(marker in href for marker in excluded_markers):
                        article_title = link.get_text().strip()
                        articles.append({
                            'url': urljoin(base_url, href),
                            'title': article_title
                        })
        
        # Look for pagination links
        pagination_urls = []
        if pages_div:
            # Find "next page" or "previous page" links
            for link in pages_div.find_all('a', href=True):
                href = link['href']
                # Pagination links contain these query parameters
                if 'pagefrom=' in href or 'pageuntil=' in href or 'pageupto=' in href:
                    pagination_urls.append(urljoin(base_url, href))
        
        return subcats, articles, category_name, pagination_urls
    
    def save_article(self, article_data, category_name, parent_categories):
        """Save article data to jsonl file."""
        normalized_url = self._normalize_url(article_data['url'])
        
        with self.lock:
            # Check if already collected
            if normalized_url in self.visited_articles:
                self.duplicates_skipped += 1
                return
            
            # Mark as visited
            self.visited_articles.add(normalized_url)
            
            # Save to file
            with open(self.output_file, 'a', encoding='utf-8') as f:
                data = {
                    'url': article_data['url'],
                    'title': article_data['title'],
                    'category': category_name,
                    'parent_categories': parent_categories
                }
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')
            self.article_count += 1
    
    def crawl(self, start_url):
        """Crawl categories in parallel."""
        to_visit = [(start_url, [])]  # (url, parent_categories)
        pbar = tqdm(desc="Crawling", unit=" cats")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while to_visit:
                # Submit batch of tasks
                futures = {}
                batch = to_visit[:self.max_workers * 2]
                to_visit = to_visit[len(batch):]
                
                for url, parents in batch:
                    normalized = self._normalize_url(url)
                    with self.lock:
                        if normalized in self.visited:
                            continue
                        self.visited.add(normalized)
                    futures[executor.submit(self.fetch_page, url)] = (url, parents)
                
                # Process results
                for future in as_completed(futures):
                    url, parents = futures[future]
                    content = future.result()
                    pbar.update(1)
                    
                    if content:
                        subcats, articles, category_name, pagination_urls = self.parse_category(content, url)
                        
                        # Build new parent path
                        new_parents = parents + [category_name]
                        
                        # Add pagination URLs to queue (treat as category pages)
                        for page_url in pagination_urls:
                            normalized = self._normalize_url(page_url)
                            with self.lock:
                                if normalized not in self.visited:
                                    self.visited.add(normalized)
                                    to_visit.append((page_url, parents))
                        
                        # Add new subcategories to queue (with deduplication)
                        for subcat in subcats:
                            normalized = self._normalize_url(subcat['url'])
                            with self.lock:
                                if normalized not in self.visited:
                                    # Don't add to visited yet - let the batch loop do it
                                    to_visit.append((subcat['url'], new_parents))
                        
                        # Save articles progressively
                        for article in articles:
                            self.save_article(article, category_name, parents)
                        
                        pbar.set_postfix({
                            'articles': self.article_count, 
                            'duplicates': self.duplicates_skipped,
                            'queue': len(to_visit)
                        })
        
        pbar.close()


if __name__ == "__main__":
    start_url = "https://ar.wikipedia.org/wiki/تصنيف:تاريخ_مصر_حسب_الحقبة"
    
    crawler = WikiCrawler(output_file='./data/articles-urls.jsonl', max_workers=3, delay=0.5)
    print("Starting crawl...")
    crawler.crawl(start_url)
    
    print(f"\n{'='*60}")
    print(f"Crawl complete! Found {crawler.article_count} unique articles")
    print(f"Results saved to: {crawler.output_file}")
    print(f"{'='*60}")
