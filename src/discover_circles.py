"""知乎AI圈子发现脚本 - 一次性获取所有符合条件的AI圈子列表"""
import re
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class ZhihuCircleDiscoverer:
    """知乎圈子发现器 - 用于一次性发现并保存所有AI相关圈子"""

    # AI相关关键词（使用完整单词匹配，需包含空格或边界）
    AI_KEYWORDS = [
        'AI ', ' AI', 'AI，', 'AI、', 'AI｜',  # 'AI' 后面跟分隔符
        '人工智能', 'artificial intelligence',
        '机器学习', '深度学习',
        'chatgpt', 'ChatGPT', 'CHATGPT',
        'gpt-', ' GPT', 'GPT ',  # GPT-4, GPTs 等
        'LLM', 'llm', ' LLM ', ' LLM｜',
        '大模型', 'aigc', 'AIGC',
        '神经网络', '自然语言处理',
        'NLP ', ' NLP ',  # NLP 需要独立
        'prompt', 'Prompt', 'PROMPT',
        'claude', 'Claude', 'CLAUDE',
        'midjourney', 'Midjourney',
        'stable diffusion', 'Stable Diffusion',
        'pytorch', 'PyTorch',
        'tensorflow', 'TensorFlow',
        'huggingface', 'HuggingFace',
        'langchain', 'LangChain',
        'openai', 'OpenAI',
        'deepseek', 'DeepSeek'
    ]

    # 排除关键词（包含这些词的圈子肯定不是AI相关）
    EXCLUDE_KEYWORDS = [
        '阅读', '书', '写作',  # 除非是 "AI写作"
        '料理', '美食', '吃', '做饭', '菜谱',
        '穿搭', '时尚', '美妆', '减肥',
        '育儿', '宝宝', '怀孕',
        '情感', '恋爱', '分手',
        '旅游', '景点', '攻略',
        '游戏', '原神', '王者荣耀',
        '追星', '粉丝', '偶像',
        '躺赚', '赚钱', '副业',
        '设计',  # 除非是 "AI设计"
        '量化交易', '股票', '基金',
        '在废纸', '书写'
    ]

    # 白名单：明确包含这些词的组合才是AI相关
    WHITELIST_PATTERNS = [
        'AI写作', 'AI设计', 'AI安全',
        'AI穿戴', 'AI Coding', 'AI工具',
        'AI时代', 'AI与人类', 'AI Hub',
        '科研AI', 'DeepSeek', 'OpenMCP'
    ]

    def __init__(self, headless=False):
        self.headless = headless
        self.page = None
        self.circles = []

    def _init_page(self):
        if self.page:
            try:
                _ = self.page.url
                return self.page
            except:
                self._close_page()

        try:
            from DrissionPage import ChromiumPage, ChromiumOptions
        except ImportError:
            print("[ERROR] 请先安装 DrissionPage: pip install DrissionPage")
            return None

        if self.headless:
            co = ChromiumOptions()
            co.set_headless(True)
            self.page = ChromiumPage(addr_driver_opts=co)
        else:
            self.page = ChromiumPage()
        return self.page

    def _close_page(self):
        if self.page:
            try:
                self.page.quit()
            except:
                pass
            self.page = None

    def _check_login(self, url):
        if not self.page:
            return
        page_text = self.page('xpath://body').text
        if "登录" in page_text and "扫码" in page_text:
            print("[INFO] 检测到登录页面，需要手动登录")
            input("扫码登录后按回车继续...")
            self.page.get(url)
            time.sleep(3)

    def _scroll_page(self, max_scrolls=30, interval=1.5):
        """滚动加载内容"""
        last_height = 0
        for i in range(max_scrolls):
            self.page.run_js('window.scrollTo(0, document.body.scrollHeight);')
            time.sleep(interval)
            current_height = self.page.run_js('return document.body.scrollHeight')
            if current_height == last_height:
                break
            last_height = current_height
            if (i + 1) % 5 == 0:
                print(f"[INFO] 滚动进度: {i+1}/{max_scrolls}")

    def _is_ai_related(self, text):
        """判断是否与AI相关"""
        if not text:
            return False

        text_original = text
        text = text.strip()

        # 先检查白名单模式（优先级最高）
        for pattern in self.WHITELIST_PATTERNS:
            if pattern in text:
                return True

        # 检查排除关键词
        for exclude_kw in self.EXCLUDE_KEYWORDS:
            if exclude_kw in text:
                # 如果同时包含白名单模式，则不排除
                for pattern in self.WHITELIST_PATTERNS:
                    if pattern in text_original:
                        return True
                return False

        # 检查AI关键词（精确匹配）
        for ai_kw in self.AI_KEYWORDS:
            if ai_kw in text:
                return True

        return False

    def _get_member_count(self):
        """获取当前圈子的成员数量"""
        # 检查是否被重定向到登录页
        current_url = self.page.url
        if 'signin' in current_url or 'login' in current_url:
            print("    [WARN] 未登录，无法获取成员数")
            return 0

        selectors = [
            '.RingHeader-memberCount',
            '.MemberCount',
            '[class*="member"]',
            '[class*="Member"]',
            'span[class*="count"]',
            'div[class*="count"]',
        ]

        for selector in selectors:
            try:
                elem = self.page.ele(f'css:{selector}', timeout=0.5)
                if elem:
                    text = elem.text.strip()
                    # 匹配 "1.2万"、"1234人"、"1,234 人" 等格式
                    match = re.search(r'([\d\,\.]+)(万)?[人名\s]*', text)
                    if match:
                        num_str = match.group(1).replace(',', '')
                        num = float(num_str)
                        if match.group(2):  # 有"万"
                            num = int(num * 10000)
                        else:
                            num = int(num)
                        return num
            except:
                continue

        # 如果找不到成员数，尝试从页面文本中提取
        try:
            page_text = self.page.text
            # 匹配 "成员 1.2万"、"1,234人加入" 等模式
            matches = re.findall(r'成员[\s\:：]*([\d\,\.]+)(万)?[人名\s]*', page_text)
            if matches:
                num_str = matches[0][0].replace(',', '')
                num = float(num_str)
                if matches[0][1]:  # 有"万"
                    num = int(num * 10000)
                else:
                    num = int(num)
                return num
        except:
            pass

        return 0

    def _get_ring_name(self):
        """获取圈子名称"""
        try:
            name_elem = self.page.ele('css:h1, .RingHeader-name', timeout=2)
            return name_elem.text.strip() if name_elem else ''
        except:
            return ''

    def discover(self, min_members=500, max_scrolls=30, force_refresh=False, skip_member_check=False):
        """发现所有AI相关圈子并过滤成员数

        Args:
            min_members: 最小成员数阈值（设为0则不过滤）
            max_scrolls: 最大滚动次数
            force_refresh: 是否强制刷新（忽略已保存的圈子列表）
            skip_member_check: 跳过成员数检查（未登录时使用）
        """
        circles_file = Path(__file__).parent.parent / 'data' / 'zhihu_ai_circles.json'

        # 如果文件存在且不强制刷新，直接加载
        if circles_file.exists() and not force_refresh:
            with open(circles_file, 'r', encoding='utf-8') as f:
                self.circles = json.load(f)
            print(f"[INFO] 从文件加载了 {len(self.circles)} 个AI圈子")
            print(f"[INFO] 文件路径: {circles_file}")
            print(f"[INFO] 如需重新发现，设置 force_refresh=True")
            return self.circles

        page = self._init_page()
        if not page:
            return []

        # 访问知乎圈子首页
        print("[INFO] 访问知乎圈子首页: https://www.zhihu.com/ring")
        page.get("https://www.zhihu.com/ring")
        time.sleep(5)
        self._check_login("https://www.zhihu.com/ring")

        # 滚动加载所有圈子
        print(f"[INFO] 滚动加载圈子列表 (最多{max_scrolls}次)...")
        self._scroll_page(max_scrolls)

        # 提取所有圈子链接
        print("\n[INFO] 提取圈子链接...")
        circles = set()
        try:
            links = page.eles('css:a')
            for link in links:
                href = link.attr('href') or ''
                if '/ring/host/' in href:
                    ring_id = href.split('/ring/host/')[-1].split('?')[0]
                    if ring_id and len(ring_id) > 5:
                        text = link.text.strip() or f"Circle_{ring_id}"
                        circles.add((ring_id, text[:200]))
        except:
            pass

        print(f"[INFO] 找到 {len(circles)} 个圈子")

        # 筛选AI相关圈子
        ai_circles = []
        for ring_id, name in circles:
            if self._is_ai_related(name):
                ai_circles.append((ring_id, name))

        print(f"[INFO] 筛选出 {len(ai_circles)} 个AI相关圈子")

        # 检查成员数（如果需要）
        self.circles = []
        if skip_member_check:
            print("[INFO] 跳过成员数检查，直接保存所有AI圈子")
            for ring_id, name in ai_circles:
                self.circles.append({
                    'ring_id': ring_id,
                    'name': name.split('\n')[0][:100],
                    'members': 0,
                    'url': f"https://www.zhihu.com/ring/host/{ring_id}"
                })
        else:
            print(f"\n[INFO] 开始检查成员数（>={min_members}人）...")
            print("-" * 70)

            for i, (ring_id, name) in enumerate(ai_circles, 1):
                url = f"https://www.zhihu.com/ring/host/{ring_id}"
                page.get(url)
                time.sleep(1)

                members = self._get_member_count()
                ring_name = self._get_ring_name() or name.split('\n')[0][:50]
                display_name = ring_name[:35] + '...' if len(ring_name) > 35 else ring_name

                if members >= min_members:
                    self.circles.append({
                        'ring_id': ring_id,
                        'name': ring_name,
                        'members': members,
                        'url': url
                    })
                    print(f"[{i:3d}] ✓ {display_name:40s} ({members:>8,}人)")
                else:
                    reason = "不足500人" if members > 0 else "无法获取"
                    print(f"[{i:3d}] ✗ {display_name:40s} ({members:>5}人，{reason})")

            print("-" * 70)

        print(f"\n[INFO] 共找到 {len(self.circles)} 个AI圈子")
        if skip_member_check:
            print("[WARNING] 成员数检查已跳过，建议后续手动确认圈子活跃度")

        # 保存结果
        self._save_circles(circles_file)
        return self.circles

    def _save_circles(self, output_file):
        """保存圈子列表"""
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.circles, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 圈子列表已保存到: {output_file}")

    def close(self):
        """关闭浏览器"""
        self._close_page()

    @staticmethod
    def load_circles():
        """静态方法：直接加载已保存的圈子列表"""
        circles_file = Path(__file__).parent.parent / 'data' / 'zhihu_ai_circles.json'
        if circles_file.exists():
            with open(circles_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []


if __name__ == '__main__':
    import sys
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')

    discoverer = ZhihuCircleDiscoverer(headless=False)

    try:
        # 方式1: 检查成员数（需要登录知乎）
        # circles = discoverer.discover(
        #     min_members=500,          # 最小成员数
        #     max_scrolls=30,           # 最大滚动次数
        #     force_refresh=True,       # 强制刷新
        #     skip_member_check=False   # 检查成员数
        # )

        # 方式2: 跳过成员数检查（未登录时使用）
        circles = discoverer.discover(
            min_members=0,            # 不过滤成员数
            max_scrolls=30,           # 最大滚动次数
            force_refresh=True,       # 强制刷新
            skip_member_check=True    # 跳过成员数检查
        )

        print(f"\n[INFO] 发现完成！共 {len(circles)} 个AI圈子")
        print("\n前10个圈子:")
        for c in circles[:10]:
            print(f"  - {c['name']}")

    finally:
        discoverer.close()
