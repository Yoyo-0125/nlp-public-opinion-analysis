import json
import os
import sys
import time
import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer
from dotenv import load_dotenv

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from src.data_crawler import ZhihuCircleCrawler
from src.models.lstm import LSTMClassifier

TARGET_POSTS = 3000
CIRCLES_FILE = 'data/zhihu_ai_circles.json'

# 自动生成带时间戳的文件名
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M')
DATA_FILE = f'data/zhihu_ring_data_{TIMESTAMP}.json'
RESULTS_FILE = f'data/zhihu_ring_data_{TIMESTAMP}_senti.json'


def load_tokenizer():
    path = os.getenv('ROBERTA_MODEL_PATH', './src/models/chinese-roberta-wwm-ext')
    return AutoTokenizer.from_pretrained(path) if os.path.exists(path) else AutoTokenizer.from_pretrained('bert-base-chinese')


def load_model(tokenizer, path='src/models/lstm_small_classifier.pth'):
    model = LSTMClassifier(tokenizer.vocab_size, 128, 64, 4, 1, tokenizer.pad_token_id)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"  模型加载完成: {path}")
    model.eval()
    return model


def predict_sentiment(model, texts, tokenizer, batch_size=32):
    predictions = []
    for i in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[i:i+batch_size], padding=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad():
            probs = torch.sigmoid(model(encoded['input_ids'], encoded['attention_mask'])).squeeze()
            preds = (probs > 0.5).long()
            predictions.extend([preds.item()] if preds.dim() == 0 else preds.tolist())
    return predictions


def analyze(data, model, tokenizer):
    print("\n[情感分析]")
    predictions = predict_sentiment(model, [d['content'] for d in data], tokenizer)
    for i, item in enumerate(data):
        item['sentiment'] = '正面' if predictions[i] >= 0.5 else '负面'
        item['sentiment_score'] = float(predictions[i])
    pos = sum(1 for p in predictions if p >= 0.5)
    print(f"  正面: {pos} 条 ({pos/len(predictions)*100:.1f}%) | 负面: {len(predictions)-pos} 条")
    return data


def crawl(circles, target):
    print(f"\n[爬取数据] 目标: {target} 条")

    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            count = len(json.load(f))
        if count >= target:
            print(f"  数据已足够: {count} 条")
            return

    crawler = ZhihuCircleCrawler(headless=False)
    try:
        for i, circle in enumerate(circles, 1):
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r', encoding='utf-8') as f:
                    total = len(json.load(f))
                if total >= target:
                    break

            print(f"  [{i}/{len(circles)}] {circle['name']}")
            try:
                crawler.crawl_ring(circle['ring_id'], max_days=0, save=True, max_posts=target)
                time.sleep(2)
            except Exception as e:
                print(f"    错误: {e}")
    finally:
        crawler.close()


def print_summary(data):
    if not data:
        return
    print(f"\n{'='*50}\n摘要")
    by_sent = {'正面': [], '负面': []}
    for d in data:
        s = d.get('sentiment', '未知')
        if s in by_sent:
            by_sent[s].append(d)
    print(f"总计: {len(data)} | 正面: {len(by_sent['正面'])} | 负面: {len(by_sent['负面'])}")

    for label, items in by_sent.items():
        if items:
            top = sorted(items, key=lambda x: x.get('likes', 0), reverse=True)[:3]
            print(f"\n{label} Top 3:")
            for j, item in enumerate(top, 1):
                print(f"  {j}. {item['content'][:60]}... (赞:{item.get('likes',0)})")


def main():
    print("=" * 50 + "\nAI观点情感分析系统")

    print("\n[1/3] 加载模型...")
    tokenizer = load_tokenizer()
    model = load_model(tokenizer)

    print("\n[2/3] 爬取数据...")
    with open(CIRCLES_FILE, 'r', encoding='utf-8') as f:
        circles = json.load(f)
    crawl(circles, TARGET_POSTS)

    print("\n[3/3] 分析...")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result = analyze(data, model, tokenizer)

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {RESULTS_FILE}")

    print_summary(result)


if __name__ == '__main__':
    main()
