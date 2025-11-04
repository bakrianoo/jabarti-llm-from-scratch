"""
Generate training and fine-tuning dataset from Egyptian figures data.
Uses OpenAI GPT-4.1-mini to create articles and question-answer pairs.

Output: JSONL file with records containing:
  - name: Figure name
  - title: Article title
  - article: Generated article for GPT training (200-1500 words, varies by importance)
  - qa_pairs: List of Q&A pairs for instruction fine-tuning
  - source_url: Wikipedia URL
  - category: Figure category
  - skip: Boolean flag (true if skipped)
  - skip_reason: Reason for skipping (if applicable)

Usage:
    python step-04-generate-train_phase2-and-finetune-dataset --input ./data/egyptian_figures_data.jsonl --output ./data/output_dataset.jsonl --max_records 50000
"""

import argparse
import json
import json_repair
from pathlib import Path
from typing import Dict, List, Set
from openai import OpenAI
from tqdm import tqdm
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from dotenv import dotenv_values

config = dotenv_values("../.env")


def load_processed_records(output_path: Path) -> Set[str]:
    """
    Load figure names that have already been processed from existing JSONL file.
    
    Returns:
        Set of figure names that already have records
    """
    if not output_path.exists():
        return set()
    
    processed = set()
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                if 'name' in record:
                    processed.add(record['name'])
        print(f"Found {len(processed)} already processed figures")
    except Exception as e:
        print(f"Warning: Could not load existing records: {e}")
        return set()
    
    return processed


def load_figures_data(input_path: Path, max_records: int = None) -> List[Dict]:
    """
    Load Egyptian figures data from JSONL file.
    
    Returns:
        List of figure records
    """
    print(f"Loading figures data from {input_path}...")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    figures = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            figures.append(record)
            
            if max_records and len(figures) >= max_records:
                break
    
    print(f"Loaded {len(figures)} figure records")
    return figures


def generate_article_and_qa(figure: Dict, client: OpenAI) -> tuple[Dict[str, any], Dict[str, int]]:
    """
    Generate article and Q&A pairs from figure data using OpenAI API.
    
    Args:
        figure: Dict with figure information (name, summary, infobox, etc.)
        client: OpenAI client instance
    
    Returns:
        Tuple of (result_dict, token_usage) where:
        - result_dict contains 'skip', 'skip_reason', 'article', 'qa_pairs' keys
        - token_usage contains 'prompt_tokens' and 'completion_tokens'
    """
    name = figure.get('name', 'Unknown')
    summary = figure.get('summary', '')
    infobox = figure.get('infobox', {})
    category = figure.get('category', '')
    
    # Prepare seed data
    seed_data = f"الاسم: {name}\n"
    if category:
        seed_data += f"التصنيف: {category}\n"
    if summary:
        seed_data += f"\nالملخص:\n{summary}\n"
    if infobox:
        seed_data += f"\nمعلومات إضافية:\n{json.dumps(infobox, ensure_ascii=False, indent=2)}\n"
    
    prompt = f"""بناءً على البيانات التالية عن "{name}"، قم بتحليل المحتوى أولاً ثم اتخاذ القرار:

البيانات الأساسية:
{seed_data}

**خطوة 1 - التحليل والقرار:**

يجب عليك تحديد ما إذا كان يجب تخطي هذا السجل (skip) أم معالجته:

**حالات التخطي (skip: true):**

1. **ليس عن شخص**: إذا كان المقال عن مكان، حدث، مؤسسة، مفهوم، أو أي شيء آخر غير شخص/شخصية
   - مثال: مسجد، مدينة، بطولة، منظمة
   - skip_reason: "not_a_person"

2. **بيانات غير كافية عن الشخص**: إذا كانت المعلومات قليلة جداً أو غامضة
   - أقل من 3 جمل مفيدة
   - معلومات سطحية فقط
   - skip_reason: "insufficient_data"

3. **غير مرتبط بمصر**: إذا كان الشخص ليس له أي علاقة بمصر
   - ملاحظة: يمكن قبول شخص غير مصري إذا كانت له علاقة بمصر (عاش فيها، عمل فيها، ساهم في تاريخها، إلخ)
   - skip_reason: "not_egypt_related"

**حالات المعالجة (skip: false):**
- المقال عن شخص/شخصية
- البيانات كافية لكتابة مقالة مفيدة
- الشخص مصري أو له علاقة واضحة بمصر

---

**خطوة 2 - إذا قررت المعالجة (skip: false):**

أولاً - عنوان المقالة:
- اكتب عنواناً جذاباً ومختصراً للمقالة (3-8 كلمات)
- يجب أن يعكس العنوان جوهر الشخصية أو إنجازها الأبرز
- مثال: "هاني عازر: مهندس برلين الحديثة" أو "نجيب محفوظ: عميد الأدب العربي"

ثانياً - المقالة:

**الأسلوب والطول:**
- الأسلوب: أكاديمي لكن مختلف عن أسلوب ويكيبيديا التقليدي
- تجنب الأسلوب الموسوعي الجاف - اكتب بطريقة أكثر سلاسة وجاذبية
- استخدم السرد القصصي عند الإمكان (storytelling)
- ابدأ بمقدمة تشد القارئ، واختم بخلاصة ملهمة أو تأملية
- استخدم اللغة العربية الفصحى الجميلة والمعاصرة

**الطول المتغير حسب أهمية الشخصية:**
- الشخصيات الأقل شهرة أو البيانات المحدودة: 200-400 كلمة
- الشخصيات المتوسطة الأهمية: 400-800 كلمة
- الشخصيات البارزة والمشهورة: 800-1500 كلمة

**إثراء المحتوى للشخصيات البارزة:**
يمكنك إضافة معلومات من معرفتك إذا كنت واثقاً 100% منها، خاصة:
- **للكُتّاب والأدباء**: أسماء أعماله الأدبية، جوائزه، أسلوبه الأدبي، تأثيره في الأدب العربي
- **للعلماء والمفكرين**: نظرياته، اكتشافاته، مؤلفاته العلمية، مساهماته البحثية
- **للفنانين**: أعماله الفنية الشهيرة، معارضه، أسلوبه الفني، تأثيره
- **للسياسيين**: مناصبه، قراراته المهمة، إنجازاته، تأثيره التاريخي
- **للرياضيين**: بطولاته، أرقامه القياسية، فرقه، إنجازاته
- **للمهندسين والمعماريين**: مشاريعه الكبرى، تصميماته، جوائزه المهنية

**المحتوى الأساسي:**
- استخدم البيانات الأساسية المقدمة كأساس
- نظّم المقالة بشكل منطقي (نشأته، إنجازاته، تأثيره، إلخ)
- اربط بين الأفكار بشكل سلس

**قواعد حيادية صارمة - مهم جداً:**
- **لا تُبدِ أي رأي شخصي** عن الشخصية (إيجابي أو سلبي)
- **التزم بالحقائق الموضوعية فقط** - لا تصدر أحكاماً تقييمية
- **تجنب الآراء السياسية أو الأيديولوجية** تماماً
- **لا تتطرق للجدل أو المواضيع الحساسة** (دينية، سياسية، عرقية)
- **تجنب الصفات الذاتية** مثل: "عظيم"، "سيء"، "الأفضل"، "الأسوأ"
- **استخدم لغة محايدة** وصفية وليست تقييمية
- مثال سيء: "كان محفوظ أعظم روائي في التاريخ" ❌
- مثال جيد: "حصل محفوظ على جائزة نوبل للآداب عام 1988" ✓

**مهم جداً - ترجمة الأسماء للعربية:**
- احرص على ترجمة الأسماء الأجنبية للعربية بشكل صحيح وفقاً للنطق الفعلي
- استخدم معرفتك بالنطق العربي الصحيح للأسماء
- أمثلة:
  * "Hani Azer" → "هاني عازر" (وليس "هاني أزر" أو "هاني آزر")
  * "Alexander" → "ألكسندر" (وليس "أليكساندر")
  * "George" → "جورج" (وليس "جيورج")
- إذا كان لديك شك في النطق الصحيح، استخدم الاسم الإنجليزي كما هو

ثالثاً - الأسئلة والأجوبة:

السؤال الأول - سؤال مباشر (Factual):
- يطلب معلومة محددة (تاريخ، مكان، اسم، رقم، حدث)
- الإجابة واضحة ومباشرة من المقالة
- يبدأ بـ: "ما"، "من"، "متى"، "أين"، "كم"
- **يجب أن يكون السؤال مكتفٍ ذاتياً ويتضمن السياق الضروري**
- مثال: "متى تأسست مدينة القاهرة؟" (وليس "متى تأسست؟")

السؤال الثاني - سؤال تحليلي (Analytical):
- يتطلب استنتاج أو تحليل من المعلومات
- يربط بين أفكار مختلفة
- يبدأ بـ: "لماذا"، "كيف"، "ما أهمية"، "ما العلاقة"، "ما الأثر"
- **يجب أن يكون السؤال مكتفٍ ذاتياً ويتضمن السياق الضروري**

قواعد الأسئلة:
- كل سؤال يجب أن يكون مفهوماً بدون الرجوع للمقالة
- اذكر الأسماء الكاملة داخل السؤال
- طول الإجابة: 1-3 جمل (20-80 كلمة)
- الإجابات مستندة للمقالة (لا تختلق معلومات)

---

**صيغة الإخراج (JSON فقط، بدون markdown):**

إذا قررت التخطي:
{{
  "skip": true,
  "skip_reason": "not_a_person" أو "insufficient_data" أو "not_egypt_related"
}}

إذا قررت المعالجة:
{{
  "skip": false,
  "title": "عنوان المقالة الجذاب",
  "article": "نص المقالة الكامل...",
  "qa_pairs": [
    {{"question": "السؤال المباشر", "answer": "الإجابة المباشرة"}},
    {{"question": "السؤال التحليلي", "answer": "الإجابة التحليلية"}}
  ]
}}"""
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "developer", "content": "أنت خبير في تحليل البيانات وكتابة مقالات تعليمية محايدة تماماً وإنشاء أسئلة عالية الجودة من المحتوى العربي. التزم بالحيادية المطلقة - لا آراء، لا تحيز، لا جدل. اتبع التعليمات بدقة."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3000,
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Extract token usage
        token_usage = {
            'prompt_tokens': completion.usage.prompt_tokens,
            'completion_tokens': completion.usage.completion_tokens
        }
        
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        # Parse JSON with repair
        result = json_repair.loads(response_text)
        
        # Validate structure
        if not isinstance(result, dict) or 'skip' not in result:
            print(f"  Warning: Invalid response format for '{name}'")
            return {}, token_usage
        
        # If skip is true, just validate skip_reason and return
        if result.get('skip', False):
            if 'skip_reason' not in result:
                print(f"  Warning: Skip without reason for '{name}'")
                result['skip_reason'] = 'unknown'
            return result, token_usage
        
        # If not skipping, validate article and qa_pairs
        if 'article' not in result or 'qa_pairs' not in result or 'title' not in result:
            print(f"  Warning: Missing title, article or qa_pairs for '{name}'")
            return {}, token_usage
        
        if not isinstance(result['qa_pairs'], list) or len(result['qa_pairs']) != 2:
            print(f"  Warning: Expected 2 QA pairs for '{name}', got {len(result.get('qa_pairs', []))}")
            return {}, token_usage
        
        # Validate article length (now accepting up to 1500 words)
        article_words = len(result['article'].split())
        if article_words < 50 or article_words > 2000:
            print(f"  Warning: Article length out of range ({article_words} words) for '{name}'")
            return {}, token_usage
        
        # Validate QA pairs
        valid_qa_pairs = []
        for qa in result['qa_pairs']:
            if not isinstance(qa, dict) or 'question' not in qa or 'answer' not in qa:
                continue
            if not qa['question'].strip() or not qa['answer'].strip():
                continue
            
            answer_words = len(qa['answer'].split())
            if answer_words < 1 or answer_words > 100:
                continue
            
            valid_qa_pairs.append(qa)
        
        if len(valid_qa_pairs) != 2:
            print(f"  Warning: Only {len(valid_qa_pairs)} valid QA pairs for '{name}'")
        
        result['qa_pairs'] = valid_qa_pairs
        return result, token_usage
        
    except Exception as e:
        print(f"  Error generating content for '{name}': {e}")
        return {}, {'prompt_tokens': 0, 'completion_tokens': 0}


def create_output_record(figure: Dict, result: Dict[str, any]) -> Dict:
    """
    Create an output record combining figure metadata with generated content.
    
    Args:
        figure: Original figure data
        result: Generated content with 'skip', 'title', 'article' and 'qa_pairs'
    
    Returns:
        Dict with complete output record
    """
    base_record = {
        "name": figure.get('name', ''),
        "url": figure.get('url', ''),
        "category": figure.get('category', '')
    }
    
    # If skipped, add skip info
    if result.get('skip', False):
        base_record['skip'] = True
        base_record['skip_reason'] = result.get('skip_reason', 'unknown')
        return base_record
    
    # Otherwise, add full content
    base_record['skip'] = False
    base_record['title'] = result.get('title', '')
    base_record['article'] = result.get('article', '')
    base_record['qa_pairs'] = result.get('qa_pairs', [])
    return base_record


def main():
    parser = argparse.ArgumentParser(description="Generate training dataset from Egyptian figures data")
    parser.add_argument('--input', type=str, default='egyptian_figures_data.jsonl', help='Input JSONL file path')
    parser.add_argument('--output', type=str, default='output_dataset.jsonl', help='Output JSONL file path')
    parser.add_argument('--max_records', type=int, default=100, help='Maximum number of records to process')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between API calls (seconds)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling')
    parser.add_argument('--workers', type=int, default=5, help='Number of concurrent workers (default: 5)')
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=config.get("OPENAI_API_KEY")
    )
    
    # Load figures
    input_path = Path(args.input)
    figures = load_figures_data(input_path, max_records=args.max_records)
    
    # Shuffle figures
    random.shuffle(figures)
    print(f"Shuffled figures with seed={args.seed}")
    
    # Setup output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load already processed figures
    processed_figures = load_processed_records(output_path)
    
    # Filter out already processed figures
    figures = [f for f in figures if f.get('name', '') not in processed_figures]
    print(f"Figures to process: {len(figures)} (skipping {len(processed_figures)} already done)")
    
    if not figures:
        print("No new figures to process!")
        return
    
    total_records = 0
    total_skipped = 0
    skip_reasons = {'not_a_person': 0, 'insufficient_data': 0, 'not_egypt_related': 0, 'unknown': 0}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    # Thread-safe lock for file writing and counter updates
    write_lock = threading.Lock()
    
    # Rate limiter (simple token bucket)
    last_request_time = [time.time()]
    rate_lock = threading.Lock()
    
    def rate_limited_generate(figure):
        """Generate with rate limiting."""
        with rate_lock:
            elapsed = time.time() - last_request_time[0]
            if elapsed < args.delay:
                time.sleep(args.delay - elapsed)
            last_request_time[0] = time.time()
        
        return generate_article_and_qa(figure, client)
    
    def process_figure(figure):
        """Process a single figure and return the result."""
        result, token_usage = rate_limited_generate(figure)
        
        if not result:
            return None, token_usage
        
        output_record = create_output_record(figure, result)
        return output_record, token_usage
    
    # Create progress bar
    pbar = tqdm(total=len(figures), desc="Processing figures")
    
    # Process figures concurrently
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_figure = {executor.submit(process_figure, figure): figure for figure in figures}
        
        # Open file for writing
        with open(output_path, 'a', encoding='utf-8') as f:
            # Process completed tasks as they finish
            for future in as_completed(future_to_figure):
                try:
                    output_record, token_usage = future.result()
                    
                    with write_lock:
                        # Update token counters
                        total_prompt_tokens += token_usage['prompt_tokens']
                        total_completion_tokens += token_usage['completion_tokens']
                        
                        # Skip if generation failed
                        if output_record:
                            # Track skip statistics
                            if output_record.get('skip', False):
                                total_skipped += 1
                                skip_reason = output_record.get('skip_reason', 'unknown')
                                skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
                            else:
                                total_records += 1
                            
                            # Write to JSONL
                            f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                            f.flush()  # Ensure data is written immediately
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            'records': total_records,
                            'skipped': total_skipped,
                            'in_tok': total_prompt_tokens,
                            'out_tok': total_completion_tokens
                        })
                
                except Exception as e:
                    print(f"\n  Error processing figure: {e}")
                    with write_lock:
                        pbar.update(1)
    
    pbar.close()
    
    total_tokens = total_prompt_tokens + total_completion_tokens
    print(f"\n✓ Generated {total_records} valid records")
    print(f"✓ Skipped {total_skipped} records:")
    for reason, count in skip_reasons.items():
        if count > 0:
            print(f"    - {reason}: {count}")
    print(f"✓ Token usage:")
    print(f"    Input tokens:  {total_prompt_tokens:,}")
    print(f"    Output tokens: {total_completion_tokens:,}")
    print(f"    Total tokens:  {total_tokens:,}")
    print(f"✓ Saved to: {output_path}")


if __name__ == "__main__":
    main()
