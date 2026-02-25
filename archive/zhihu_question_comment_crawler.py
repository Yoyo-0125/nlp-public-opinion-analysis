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
from pathlib import Path

load_dotenv()


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
    """微博文本搜索爬虫 - 纯 requests 方式"""

    def __init__(self):
        super().__init__()
        self.results = []
        self.url = 'https://m.weibo.cn/api/container/getIndex'
        self.params = {
            'containerid': None,
            'page_type': 'searchall',
            'page': 0
        }

    def crawl(self, keyword, max_pages=10, sleep_range=(2, 4)):
        """
        搜索微博内容

        注意：需要在 .env 中配置有效的 WEIBO_COOKIES 和 WEIBO_X_XSRF_TOKEN
        如果报错 432，说明 Cookie 过期，需要更新
        """
        encoded_keyword = quote(keyword)
        self.params['containerid'] = f'100103type=1&q={encoded_keyword}'

        print(f"[INFO] 搜索关键词: {keyword}")

        for page in tqdm(range(1, max_pages + 1)):
            self.params['page'] = page

            try:
                resp = requests.get(self.url, headers=self.HEADERS, params=self.params, timeout=10)

                if resp.status_code == 432:
                    print('\n[ERROR] 触发 432 反爬，请更新 .env 中的 WEIBO_COOKIES。')
                    break
                if resp.status_code != 200:
                    print(f'\n[ERROR] 状态码: {resp.status_code}')
                    break

                data = resp.json()

                if data.get('ok') != 1:
                    print(f'\n[INFO] 第 {page} 页无数据，抓取结束。')
                    break

                cards = data.get('data', {}).get('cards', [])

                for card in cards:
                    if card.get('mblog'):
                        self.results.append(self.make_std_json(card['mblog'], encoded_keyword))

                    elif card.get('card_group'):
                        for item in card.get('card_group'):
                            if item.get('mblog'):
                                self.results.append(self.make_std_json(item['mblog'], encoded_keyword))

                sleep(random.uniform(*sleep_range))    # 随机延时防封

            except Exception as e:
                print(f'\n[Exception] {e}')
                break

        return self.results


class WeiboHotCrawler(Crawler):
    def __init__(self):
        super().__init__()
        self.results = []
        self.url = 'https://weibo.com/ajax/side/hotSearch'

    def crawl(self):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.weibo.com/hot/search",
            }
            resp = requests.get(self.url, headers=headers)
            self.check_status_code(resp.status_code)
            data = resp.json()

            for item in data["data"]["realtime"]:
                self.results.append({
                    "rank": item.get("rank"),
                    "word": item.get("word"),
                    "label_name": item.get("label_name")
                })

        except Exception as e:
            print(f'\n[Exception] {e}')

        return self.results


class ZhihuSearchCrawler:
    """知乎搜索爬虫 - 使用 DrissionPage"""

    def __init__(self):
        self.results = []
        self.auth_file = Path("./zhihu_auth.json")

    def crawl(self, keyword, max_results=20, headless=False):
        """
        搜索知乎内容

        Args:
            keyword: 搜索关键词
            max_results: 最大结果数
            headless: 是否无头模式
        """
        try:
            from DrissionPage import ChromiumPage
        except ImportError:
            print("[ERROR] 请先安装 DrissionPage: pip install DrissionPage")
            return []

        print(f"[INFO] 启动浏览器搜索: {keyword}")

        # DrissionPage 启动浏览器
        if headless:
            from DrissionPage import ChromiumOptions
            co = ChromiumOptions()
            co.set_headless(True)
            page = ChromiumPage(addr_driver_opts=co)
        else:
            page = ChromiumPage()

        # 构造搜索 URL
        url = f"https://www.zhihu.com/search?type=content&q={quote(keyword)}"
        page.get(url)

        # 等待页面加载
        sleep(3)

        # 检查是否需要登录
        page_text = page('xpath://body').text
        page_html = page.html

        if "登录" in page_text and "SignFlowButton" in page_html:
            print("[INFO] 检测到登录页面，需要手动登录")
            print("请在浏览器中扫码登录，登录后按回车继续...")
            input()

            # 保存登录状态
            print("[INFO] 登录状态已保存")
            sleep(2)

        # 解析搜索结果
        cards = page.eles('css:div.SearchResult-Card')
        print(f"[INFO] 找到 {len(cards)} 条搜索结果")

        for i, card in enumerate(cards[:max_results]):
            try:
                # 获取卡片文本并解析
                full_text = card.text
                lines = full_text.split('\n')

                # 第一行通常是标题
                title = lines[0].strip() if lines else ""

                # 提取点赞数
                likes = 0
                for line in lines:
                    if "赞同" in line:
                        import re
                        match = re.search(r'(\d+)', line)
                        if match:
                            likes = int(match.group(1))
                            if "万" in line:
                                likes *= 10000
                        break

                # 提取作者（如果有）
                author = ""
                author_elem = card.ele('css:a.AuthorInfo-name', timeout=0.1)
                if author_elem:
                    author = author_elem.text

                self.results.append({
                    'keyword': keyword,
                    'title': title,
                    'content': full_text[:500],
                    'author': author,
                    'likes': likes,
                    'source': 'zhihu'
                })

            except Exception as e:
                print(f"[WARNING] 解析第 {i+1} 条结果失败: {e}")
                continue

        # 关闭浏览器
        page.quit()

        return self.results


if __name__ == '__main__':
    print(f"{'='*20} Testing WeiboHotCrawler {'='*20}")
    hotcrawler = WeiboHotCrawler()
    res = hotcrawler.crawl()
    print(res[:3] if res else [], len(res))

    print(f"\n{'='*20} Testing WeiboTextCrawler {'='*21}")
    textcrawler = WeiboTextCrawler()
    res = textcrawler.crawl(keyword='测试', max_pages=1)
    print(res[:2] if res else [], len(textcrawler.results))

    print(f"\n{'='*20} Testing ZhihuSearchCrawler {'='*20}")
    zhihu_crawler = ZhihuSearchCrawler()
    res = zhihu_crawler.crawl('电影', max_results=5)
    print(f"获取 {len(res)} 条结果")
    for r in res[:2]:
        print(f"  - {r['title']}")
        print(f"    点赞: {r['likes']}")
