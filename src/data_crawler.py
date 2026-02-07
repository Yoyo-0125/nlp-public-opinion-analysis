import re
import os
import json
import random
import requests
from tqdm import tqdm
from time import sleep
from urllib.parse import quote
from dotenv import load_dotenv
from fake_useragent import UserAgent

class Crawler():
    def __init__(self):
        headers_json = os.getenv('WEIBO_HEADERS')
        ua = UserAgent()
        self.HEADERS = json.loads(headers_json) if headers_json else {}
        self.HEADERS['User-Agent'] = ua.random
        self.HEADERS['Cookie'] = os.getenv('WEIBO_COOKIES')
        self.HEADERS['X-XSRF-TOKEN'] = os.getenv('WEIBO_X_XSRF_TOKEN')

    def clean_html(self, raw_html):
        # 清除文本中的 HTML 标签
        if not raw_html:
            return ''
        clean_re = re.compile('<.*?>')
        return re.sub(clean_re, '', raw_html)
    
    def make_std_json(self, mblog, keyword):
        # 提取单条微博的字段数据
        return {
            'source': 'weibo',
            'topic': keyword,
            'weibo_id': mblog.get('id'),
            'text': self.clean_html(mblog.get('text', '')),
            'url': f"https://weibo.com/{mblog.get('user', {}).get('id')}/{mblog.get('id')}",
            'timestamp': mblog.get('created_at'),
            'reposts': mblog.get('reposts_count'),
            'comments': mblog.get('comments_count'),
            'likes': mblog.get('attitudes_count')
        }
    
    def check_status_code(self, status_code):
        if status_code == 432:
            print('\n[ERROR] 触发 432 反爬，请检查 Cookie 或 X-XSRF-TOKEN。')
        if status_code != 200:
            print(f'\n[ERROR] 状态码: {status_code}')
        return 0
    

class WeiboTextCrawler(Crawler):
    def __init__(self, keyword):
        super().__init__()
        self.results = []
        self.encoded_keyword = quote(keyword)
        self.containerid = f'100103type=1&q={self.encoded_keyword}'
        self.url = 'https://m.weibo.cn/api/container/getIndex'
        self.params = {
            'containerid': self.containerid,
            'page_type': 'searchall',
            'page': 0
        }
    
    def crawl(self, max_pages=10, sleep_range=(1,3)):
        for page in tqdm(range(1, max_pages + 1)):
            self.params['page'] = page

            try:
                resp = requests.get(self.url, headers=self.HEADERS, params=self.params, timeout=10)
                self.check_status_code(resp.status_code)

                data = resp.json()
                if data.get('ok') != 1:
                    print(f'\n[INFO] 第 {page} 页无数据，抓取结束。')
                    break

                cards = data.get('data', {}).get('cards', [])
                page_count = 0

                for card in cards:
                    if card.get('mblog'):
                        self.results.append(self.make_std_json(card['mblog'], self.encoded_keyword))
                        page_count += 1
                    
                    elif card.get('card_group'):
                        for item in card.get('card_group'):
                            if item.get('mblog'):
                                self.results.append(self.make_std_json(item['mblog'], self.encoded_keyword))
                                page_count += 1
                
                sleep(random.uniform(*sleep_range))    # 随机延时防封

            except Exception as e:
                print(f'\n[Exception] {e}')
                break

        return self.results

class WeiboHotCrawler(Crawler):
    def __init__(self):
        super().__init__()
        self.results = []
        self.url = 'https://weibo.com/ajax/side/hotSearch'  # 修正为正确的热搜 API 地址

    def crawl(self):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.weibo.com/hot/search",
            } # 用这个headers更稳一点
            resp = requests.get(self.url, headers=headers)
            data = resp.json()

            for item in data["data"]["realtime"]:
                self.results.append({
                    "rank": item.get("rank"),
                    "word": item.get("word"),
                    "label_name": item.get("label_name")
                })

        except Exception as e:
            print(f'\n[Exception] {e}')
            print("status:", resp.status_code)
            print("text:", resp.text[:500])

        return self.results

if __name__ == '__main__':
    load_dotenv()
    print(f"{'='*20}Testing WeiboTextCrawler{'='*20}")
    textcrawler = WeiboTextCrawler('测试')
    res = textcrawler.crawl(max_pages=1)
    print(res[:2], len(textcrawler.results))
    
    print(f"\n{'='*20}Testing WeiboHotCrawler{'='*21}")
    hotcrawler = WeiboHotCrawler()
    res = hotcrawler.crawl()
    print(res[:3], len(res))
