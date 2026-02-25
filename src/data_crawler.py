import re
import os
import json
from time import sleep
from urllib.parse import quote
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from fake_useragent import UserAgent

load_dotenv()


class WeiboCrawler:
    def __init__(self):
        ua = UserAgent()
        self.HEADERS = json.loads(os.getenv('WEIBO_HEADERS'))
        self.HEADERS['User-Agent'] = ua.random
        self.HEADERS['Cookie'] = os.getenv('WEIBO_COOKIES', '')
        self.HEADERS['X-XSRF-TOKEN'] = os.getenv('WEIBO_X_XSRF_TOKEN', '')

    def clean_html(self, raw_html):
        """清理HTML标签"""
        if not raw_html:
            return ''
        return re.sub('<.*?>', '', raw_html)


class ZhihuCrawler:
    def __init__(self):
        self.page = None
        self.CHROME_DATA_DIR = Path(__file__).parent.parent / 'chrome_data_zhihu_ring'

    def _init_page(self):
        if self.page:
            try:
                self.page.url
                return
            except:
                pass
        from DrissionPage import ChromiumPage, ChromiumOptions
        co = ChromiumOptions()
        co.set_user_data_path(str(self.CHROME_DATA_DIR))
        co.set_argument('--no-first-run', '--no-default-browser-check')
        self.page = ChromiumPage(addr_or_opts=co)
        sleep(5)

    def close(self):
        if self.page:
            self.page.quit()
            self.page = None


class WeiboHotCrawler(WeiboCrawler):
    def __init__(self):
        super().__init__()
        self.results = []
        self.url = 'https://weibo.com/ajax/side/hotSearch'

    def crawl(self):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.weibo.com/hot/search"
            }
            resp = __import__('requests').get(self.url, headers=headers)
            data = resp.json()

            for item in data.get("data", {}).get("realtime", []):
                self.results.append({
                    "rank": item.get("rank"),
                    "word": item.get("word"),
                    "label_name": item.get("label_name")
                })
        except Exception as e:
            print(f'[ERROR] {e}')

        return self.results


class WeiboTextCrawler(WeiboCrawler):
    def __init__(self):
        super().__init__()
        self.results = []
        self.url = 'https://m.weibo.cn/api/container/getIndex'
        self.headers = {
            'User-Agent': 'Mozilla/5.0',
            'Cookie': os.getenv('WEIBO_COOKIES', ''),
            'X-XSRF-TOKEN': os.getenv('WEIBO_X_XSRF_TOKEN', '')
        }

    def crawl(self, keyword, max_pages=10):
        params = {'containerid': f'100103type=1&q={quote(keyword)}', 'page_type': 'searchall', 'page': 0}
        print(f"[INFO] 搜索关键词: {keyword}")

        requests = __import__('requests')
        random = __import__('random').uniform

        for page in range(1, max_pages + 1):
            params['page'] = page
            resp = requests.get(self.url, headers=self.headers, params=params, timeout=10)

            if resp.status_code == 432:
                print('[ERROR] Cookie过期')
                break
            if resp.status_code != 200:
                break

            data = resp.json()
            if data.get('ok') != 1:
                break

            for card in data.get('data', {}).get('cards', []):
                for mblog in [card.get('mblog')] + [item.get('mblog') for item in card.get('card_group', [])]:
                    if mblog:
                        self.results.append({
                            'source': 'weibo',
                            'weibo_id': mblog.get('id'),
                            'text': re.sub('<.*?>', '', mblog.get('text', '')),
                            'timestamp': mblog.get('created_at'),
                            'comments': mblog.get('comments_count'),
                            'likes': mblog.get('attitudes_count')
                        })
            sleep(random(2, 4))

        return self.results


class ZhihuCircleCrawler(ZhihuCrawler):
    def crawl_ring(self, ring_id, max_days=0, save=True, max_posts=9999, min_comments=0):
        """
        爬取指定圈子

        Args:
            ring_id: 圈子ID
            max_days: 爬取最近N天的帖子 (0=全部)
            save: 是否保存到文件
            max_posts: 最多爬取帖子数
            min_comments: 最少评论数筛选
        """
        self._init_page()
        url = f"https://www.zhihu.com/ring/host/{ring_id}"
        print(f"[INFO] 访问圈子: {url}")
        self.page.get(url)
        sleep(10)

        # 点击"最新"标签
        self.page.run_js('''
            const tabs = document.querySelectorAll('a');
            for (let t of tabs) {
                if (t.textContent === "最新") { t.click(); return; }
            }
        ''')
        sleep(2)

        # 滚动加载
        print("[INFO] 滚动加载帖子...")
        for i in range(30):
            self.page.run_js('window.scrollTo(0, document.body.scrollHeight)')
            sleep(0.8)
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/30")

        # 提取帖子
        posts = self._extract_posts()

        # 按赞同数排序，筛选
        posts.sort(key=lambda x: x.get('likes', 0), reverse=True)
        if min_comments > 0:
            posts = [p for p in posts if p.get('comment_count', 0) >= min_comments]
        posts = posts[:max_posts]

        print(f"[INFO] 筛选后: {len(posts)} 个帖子")

        # 爬取每个帖子的评论
        results = []
        for i, post in enumerate(posts, 1):
            print(f"[{i}/{len(posts)}] {post.get('title', '无标题')[:30]}")
            comments = self._get_post_comments(post['url'])
            results.append({
                'source': 'zhihu_circle',
                'ring_id': ring_id,
                'content': post.get('content', ''),
                'likes': post.get('likes', 0),
                'pub_time': post.get('pub_time', ''),
                'comments': comments
            })

        if save:
            self._save(results)
        return results

    def _extract_posts(self):
        """提取帖子列表"""
        return self.page.run_js('''
            const posts = [];
            const items = document.querySelectorAll('div[class*="ContentItem"]');
            items.forEach(item => {
                const contentElem = item.querySelector('div[class*="RichContent"]');
                if (!contentElem) return;

                // 获取正文
                const text = contentElem.textContent.trim();

                // 获取时间
                let pubTime = '';
                const timeElem = item.querySelector('[class*="Time"]');
                if (timeElem) pubTime = timeElem.textContent.trim();

                // 获取赞同数
                let likes = 0;
                const voteBtn = item.querySelector('button[class*="VoteButton"]');
                if (voteBtn) {
                    const match = voteBtn.ariaLabel?.match(/(\\d+)/);
                    if (match) likes = parseInt(match[1]);
                }

                // 获取评论数
                let commentCount = 0;
                const links = item.querySelectorAll('a');
                for (let a of links) {
                    const match = a.textContent.match(/(\\d+)\\s*条?\\s*评论/);
                    if (match) {
                        commentCount = parseInt(match[1]);
                        break;
                    }
                }

                // 获取帖子链接
                let url = '';
                const link = item.querySelector('a[href*="/pin/"]');
                if (link) url = link.href;

                if (text.length > 50) {
                    posts.push({
                        title: text.substring(0, 100),
                        content: text,
                        url: url,
                        likes: likes,
                        comment_count: commentCount,
                        pub_time: pubTime
                    });
                }
            });
            return posts;
        ''')

    def _get_post_comments(self, post_url):
        """获取帖子评论"""
        if not post_url:
            return []

        self.page.get(post_url)
        sleep(5)

        # 点击评论按钮
        clicked = self.page.run_js('''
            const btns = document.querySelectorAll('button');
            for (let b of btns) {
                if (b.textContent.includes('评论') || b.textContent.includes('条评论')) {
                    b.click();
                    return true;
                }
            }
            return false;
        ''')

        if not clicked:
            return []

        sleep(3)

        # 加载更多评论
        self._load_comments_more()

        # 提取评论
        comments = self.page.run_js('''
            const list = [];
            const elems = document.querySelectorAll('div[class*="CommentContent"]');
            elems.forEach(e => {
                const t = e.textContent.trim();
                if (t.length > 5) list.push(t);
            });
            return list;
        ''')

        # 关闭弹窗
        self.page.run_js('document.dispatchEvent(new KeyboardEvent("keydown", {key: "Escape"}))')
        sleep(1)

        return comments

    def _load_comments_more(self):
        """加载更多评论"""
        # 点击展开按钮
        for _ in range(3):
            self.page.run_js('''
                const btns = document.querySelectorAll('button');
                for (let b of btns) {
                    if (b.textContent.includes('查看') || b.textContent.includes('展开') || b.textContent.includes('全部')) {
                        if (b.offsetParent) b.click();
                    }
                }
            ''')
            sleep(0.5)

        # 滚动加载
        for _ in range(20):
            self.page.run_js('''
                const modal = document.querySelector('div[class*="Modal"]');
                if (modal) {
                    modal.scrollTop = modal.scrollHeight;
                }
            ''')
            sleep(0.3)

            # 滚动时点击展开
            self.page.run_js('''
                const btns = document.querySelectorAll('button');
                for (let b of btns) {
                    if (b.textContent.includes('展开') && b.offsetParent) b.click();
                }
            ''')

    def _save(self, results):
        """保存结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output = Path(__file__).parent.parent / 'data' / f'zhihu_ring_data_{timestamp}.json'

        # 合并已有数据
        existing = []
        if output.exists():
            with open(output, 'r', encoding='utf-8') as f:
                existing = json.load(f)

        # 去重 (用内容前50字符)
        seen = {e['content'][:50] for e in existing}
        new_data = [r for r in results if r['content'][:50] not in seen]

        all_data = existing + new_data
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        print(f"[INFO] 已保存: {output} (总数: {len(all_data)})")

if __name__ == '__main__':
    print("=" * 50)
    print("数据爬虫测试")
    print("=" * 50)

    print("\n=== WeiboHotCrawler ===")
    crawler1 = WeiboHotCrawler()
    results1 = crawler1.crawl()
    print(f"微博热榜: {len(results1)} 条")
    if results1:
        print(f"  第1名: {results1[0]}")

    print("\n=== WeiboTextCrawler ===")
    print("(跳过 - 需要Cookie)")

    print("\n=== ZhihuCircleCrawler ===")
    crawler3 = ZhihuCircleCrawler()
    results3 = crawler3.crawl_ring("1913608407048511547", max_posts=2, save=False)
    print(f"帖子数: {len(results3)}")
    for i, r in enumerate(results3, 1):
        print(f"  {i}. 评论{len(r['comments'])}条, 赞{r['likes']}个")
    crawler3.close()

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)
