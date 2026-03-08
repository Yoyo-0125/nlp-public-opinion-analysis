"""
BERTopic 动态主题追踪实验（升级版）
"""

import argparse
import json
import math
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import jieba
import jieba.posseg as pseg
import matplotlib
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

try:
    from hdbscan import HDBSCAN
except Exception:
    HDBSCAN = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore")

try:
    from tqdm.auto import tqdm

    tqdm.pandas()
except Exception:
    pass

LOG_FILE = Path(__file__).parent.parent / "outputs" / "run_log.txt"


class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    DATA_FILE = DATA_DIR / "zhihu_ring_data_20260225_senti.json"
    CIRCLES_FILE = DATA_DIR / "zhihu_ai_circles.json"
    MODEL_PATH = PROJECT_ROOT / "src" / "models" / "text2vec-base-chinese"

    UMAP_N_NEIGHBORS = 15
    UMAP_N_COMPONENTS = 15
    UMAP_MIN_DIST = 0.0
    UMAP_METRIC = "cosine"
    UMAP_RANDOM_STATE = 42

    N_CLUSTERS = 20
    RANDOM_STATE = 42
    MAX_FEATURES = 2500

    TOPIC_MATCH_THRESHOLD = 0.35
    TOPIC_SPLIT_THRESHOLD = 0.45

    PHRASE_MIN_COUNT = 2
    PHRASE_TOP_K = 5
    KEYWORD_TOP_K = 8
    MANUAL_AUDIT_SAMPLE_SIZE = 20
    EPS = 1e-12
    TOPIC_INFO_PRINT_ROWS = 8
    DOC_DEDUP_SIM_THRESHOLD = 0.85

    # 只允许少量“事件动词”进入关键词
    VERB_WHITELIST = {
        "爆火",
        "涨价",
        "发布",
        "开源",
        "上线",
        "集成",
        "收购",
        "融资",
        "裁员",
        "升级",
        "降价",
        "停止",
        "支持",
    }

    # 模板化表达黑名单（用于关键词与短语过滤）
    GENERIC_PHRASES = {
        "关键 结论",
        "论文 详解",
        "详解 核心",
        "核心 要点",
        "一句话",
        "推荐 清单",
        "大家 觉得",
        "不妥 之处",
        "如同意 转发",
        "如不同意",
        "拍砖",
        "有没有 兴趣",
        "深度 体验",
        "欢迎 交流",
        "以上 内容",
    }

    STOP_WORDS = {
        "的",
        "了",
        "在",
        "是",
        "我",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "一个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
        "那",
        "与",
        "对于",
        "为了",
        "因为",
        "所以",
        "但是",
        "虽然",
        "可以",
        "这个",
        "那个",
        "什么",
        "怎么",
        "如何",
        "阅读",
        "全文",
        "链接",
        "分享",
        "觉得",
        "感觉",
        "其实",
        "如果",
        "只要",
        "然后",
        "最后",
        "比如",
        "像",
        "还是",
        "或者",
        "而且",
        "不过",
        "当然",
        "可能",
        " ",
        "",
        "ai",
        "AI",
        "aI",
        "Ai",
        "模型",
        "系统",
        "方法",
        "技术",
        "数据",
        "算法",
        "问题",
        "使用",
        "需要",
        "进行",
        "通过",
        "实现",
        "基于",
        "训练",
        "学习",
        "提升",
        "优化",
        "效果",
        "结果",
        "研究",
        "分析",
        "设计",
        "开发",
        "应用",
        "工具",
        "平台",
        "用户",
        "时间",
        "信息",
        "内容",
        "文章",
        "视频",
        "帖子",
        "讨论",
        "话题",
        "大家",
        "现在",
        "已经",
        "非常",
        "功能",
        "能力",
        "场景",
        "领域",
        "行业",
        "市场",
        "企业",
        "公司",
        "团队",
        "产品",
        "服务",
        "项目",
        "LLM",
        "llm",
        "LLMs",
        "GPT",
        "gpt",
        "OpenAI",
        "openai",
        "人工智能",
        "机器学习",
        "深度学习",
        "神经网络",
        "大语言模型",
        "大模型",
        "智能",
        "自动化",
        "生成式",
        "AGI",
        "agi",
        "做",
        "用",
        "来",
        "想",
        "知道",
        "认为",
        "发现",
        "开始",
        "尝试",
        "制作",
        "篇",
        "个",
        "种",
        "次",
        "些",
        "点",
        "位",
        "条",
        "项",
        "名",
    }


def ensure_output_dir():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log(message: str):
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(message.encode("gbk", errors="replace").decode("gbk"), flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


class POSTokenizer:
    def __init__(self):
        # 收紧词性：名词/专名/英文实体为主
        self.valid_pos = {"n", "nr", "ns", "nt", "nz", "vn", "eng", "x", "nw", "nx"}
        self.verb_pos = {"v"}

    def __call__(self, text):
        words_with_pos = pseg.lcut(text)
        out = []
        for word, pos in words_with_pos:
            word = word.strip()
            if not word:
                continue
            if word in Config.STOP_WORDS or word.isdigit():
                continue

            has_latin = bool(re.search(r"[A-Za-z]", word))
            has_cjk = bool(re.search(r"[\u4e00-\u9fff]", word))

            # 英文实体可更短（如 GLM），中文仍保持 >=2
            if has_latin and len(word) < 2:
                continue
            if has_cjk and len(word) < 2:
                continue

            # 仅允许白名单事件动词进入（如“爆火/涨价/发布”）
            if pos in self.verb_pos and word not in Config.VERB_WHITELIST:
                continue

            if pos in self.valid_pos or pos in self.verb_pos or has_latin:
                out.append(word)
        return out


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"#[^#\s]+#?", "", text)
    return text.strip()


def chinese_tokenize(text):
    if not isinstance(text, str) or len(text) < 2:
        return []
    words = jieba.lcut(text)
    return [w for w in words if w not in Config.STOP_WORDS and len(w) > 1]


def load_data():
    with open(Config.DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(Config.CIRCLES_FILE, "r", encoding="utf-8") as f:
        circles_data = json.load(f)

    ring_to_name = {c["ring_id"]: c["name"] for c in circles_data}
    df = pd.DataFrame(data)
    df["ring_name"] = df["ring_id"].map(ring_to_name).fillna(df["ring_id"])

    raw_len = len(df)
    df["pub_time"] = pd.to_datetime(df["pub_time"], errors="coerce")
    bad_time = int(df["pub_time"].isna().sum())
    if bad_time:
        log(f"[WARN] 无法解析时间: {bad_time} 条，已剔除")
        df = df.dropna(subset=["pub_time"])

    log(f"数据量: {len(df)} 条 (原始 {raw_len} 条)")
    if len(df) > 0:
        log(f"时间范围: {df['pub_time'].min()} - {df['pub_time'].max()}")
    return df.reset_index(drop=True)


def build_time_bins(df, granularity: str, min_docs_per_bin: int):
    df = df.copy()
    if granularity == "month":
        df["time_bin"] = df["pub_time"].dt.to_period("M").dt.to_timestamp()
    elif granularity == "week":
        df["time_bin"] = df["pub_time"].dt.to_period("W-MON").apply(lambda p: p.start_time)
    elif granularity == "2week":
        base = df["pub_time"].min().floor("D")
        delta = ((df["pub_time"] - base).dt.days // 14).astype(int)
        df["time_bin"] = base + pd.to_timedelta(delta * 14, unit="D")
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    bin_counts = df["time_bin"].value_counts().sort_index()
    keep_bins = bin_counts[bin_counts >= min_docs_per_bin].index
    if len(keep_bins) == 0:
        log("[WARN] 所有分箱低于 min_docs_per_bin，降级为不过滤")
        out = df
    else:
        out = df[df["time_bin"].isin(keep_bins)].copy()
    log(
        "时间分箱: "
        f"granularity={granularity}, bins={out['time_bin'].nunique()}, docs={len(out)}"
    )
    if out["time_bin"].nunique() < 2:
        log("[WARN] 时间分箱不足 2 个，动态趋势可信度有限")
    return out.reset_index(drop=True)


def preprocess_data(df):
    df = df.copy()
    df["content_clean"] = df["content"].apply(clean_text)
    df["content_tokenized"] = df["content_clean"].apply(lambda x: " ".join(chinese_tokenize(x)))

    before = len(df)
    df = df[df["content_tokenized"].str.len() > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        log(f"[WARN] 空文本剔除: {dropped} 条")

    docs = df["content_clean"].tolist()
    docs_tokenized = df["content_tokenized"].tolist()
    timestamps = df["time_bin"].tolist()

    avg_len = np.mean([len(d) for d in docs]) if docs else 0
    log(f"有效文档数: {len(docs)} | 平均长度: {avg_len:.0f}")
    return df, docs, docs_tokenized, timestamps


def load_embedding_model():
    if Config.MODEL_PATH.exists():
        log(f"加载本地嵌入模型: {Config.MODEL_PATH}")
        return SentenceTransformer(str(Config.MODEL_PATH))
    log("[WARN] 本地模型不存在，降级使用在线模型")
    return SentenceTransformer("shibing624/text2vec-base-chinese")


def generate_embeddings(embedding_model, docs):
    embeddings = embedding_model.encode(docs, show_progress_bar=False, batch_size=64)
    log(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def create_vectorizer():
    return CountVectorizer(
        max_features=Config.MAX_FEATURES,
        tokenizer=POSTokenizer(),
        ngram_range=(1, 3),  # 短语优先，支持 bigram/trigram
        lowercase=False,  # 保留英文大小写（如 GLM）
        min_df=2,
        max_df=0.8,
    )


def create_umap_model():
    return UMAP(
        n_neighbors=Config.UMAP_N_NEIGHBORS,
        n_components=Config.UMAP_N_COMPONENTS,
        min_dist=Config.UMAP_MIN_DIST,
        metric=Config.UMAP_METRIC,
        random_state=Config.UMAP_RANDOM_STATE,
    )


def create_cluster_model(pipeline: str):
    if pipeline == "research" and HDBSCAN is not None:
        log("研究管线: HDBSCAN")
        return HDBSCAN(
            min_cluster_size=25,
            min_samples=10,
            metric="euclidean",
            prediction_data=True,
        )
    if pipeline == "research" and HDBSCAN is None:
        log("[WARN] hdbscan 缺失，研究管线降级为 KMeans")
    log("稳定管线: KMeans")
    return KMeans(
        n_clusters=Config.N_CLUSTERS,
        random_state=Config.RANDOM_STATE,
        max_iter=300,
        n_init=10,
    )


def train_model(embedding_model, vectorizer_model, docs_tokenized, embeddings, pipeline: str):
    cluster_model = create_cluster_model(pipeline)
    umap_model = create_umap_model() if pipeline == "research" else None
    nr_topics = "auto" if pipeline == "research" and HDBSCAN is not None else Config.N_CLUSTERS

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        representation_model=[KeyBERTInspired(), MaximalMarginalRelevance(diversity=0.3)],
        verbose=False,
        nr_topics=nr_topics,
    )
    topics, probs = topic_model.fit_transform(docs_tokenized, embeddings=embeddings)
    log(f"主题数: {len(set(topics))}")
    return topic_model, topics, probs


def load_model_for_eval(docs_tokenized, embeddings):
    model_path = Config.OUTPUT_DIR / "bertopic_model"
    if not model_path.exists():
        raise FileNotFoundError(f"eval-only 需要模型: {model_path}")
    topic_model = BERTopic.load(str(model_path))
    try:
        topics, probs = topic_model.transform(docs_tokenized, embeddings=embeddings)
    except Exception:
        topics, probs = topic_model.transform(docs_tokenized)
    return topic_model, topics, probs


def _compact_name(name: str, max_len: int = 36):
    if not isinstance(name, str):
        return ""
    return name if len(name) <= max_len else name[: max_len - 3] + "..."


def show_topic_info(topic_model, rows=8):
    """控制台仅输出精简主题摘要，避免长关键词刷屏。"""
    topic_info = topic_model.get_topic_info()
    valid = topic_info[topic_info["Topic"] != -1].copy()

    if len(valid) == 0:
        log("主题摘要: 无有效主题")
        return topic_info

    top_rows = valid.sort_values("Count", ascending=False).head(rows)
    total_docs = int(valid["Count"].sum())
    log(f"主题摘要: {len(valid)} 个有效主题, 覆盖文档 {total_docs}")
    log("Top主题 (Topic | Count | Label)")
    for _, r in top_rows.iterrows():
        topic_id = int(r["Topic"])
        count = int(r["Count"])
        words = safe_topic_words(topic_model, topic_id, top_k=3)
        if words:
            label = " / ".join(words)
        else:
            label = _compact_name(str(r.get("Name", "")))
        log(f"- {topic_id:>3} | {count:>4} | {label}")
    return topic_info


def apply_smoothing(topics_over_time: pd.DataFrame, method: str):
    out = topics_over_time.copy().sort_values(["Topic", "Timestamp"]).reset_index(drop=True)
    out["FrequencyRaw"] = out["Frequency"].astype(float)

    if method == "none":
        out["FrequencySmoothed"] = out["FrequencyRaw"]
    elif method == "rolling":
        out["FrequencySmoothed"] = out.groupby("Topic")["FrequencyRaw"].transform(
            lambda s: s.rolling(window=3, min_periods=1).mean()
        )
    elif method == "ewm":
        out["FrequencySmoothed"] = out.groupby("Topic")["FrequencyRaw"].transform(
            lambda s: s.ewm(alpha=0.4, adjust=False).mean()
        )
    else:
        raise ValueError(f"Unknown smoothing: {method}")

    # 保持与 BERTopic 可视化兼容
    out["Frequency"] = out["FrequencySmoothed"]
    return out


def compute_topics_over_time(topic_model, docs_tokenized, timestamps, smoothing: str):
    try:
        nr_bins = len(pd.Series(timestamps).dropna().unique())
        tot = topic_model.topics_over_time(
            docs_tokenized,
            timestamps,
            nr_bins=nr_bins,
            global_tuning=True,
            evolution_tuning=True,
        )
        tot = apply_smoothing(tot, smoothing)
        log(f"topics_over_time 行数: {len(tot)}")
        return tot
    except Exception as e:
        log(f"[WARN] topics_over_time 失败，降级跳过: {e}")
        traceback.print_exc()
        return None


def plot_topic_evolution(topic_model, topics_over_time, output_dir):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    topic_info = topic_model.get_topic_info()
    major_topics = (
        topic_info[topic_info["Topic"] != -1]
        .sort_values("Count", ascending=False)["Topic"]
        .head(10)
        .tolist()
    )
    data = topics_over_time[topics_over_time["Topic"].isin(major_topics)].copy()

    fig, ax = plt.subplots(figsize=(14, 8))
    for topic in major_topics:
        tdf = data[data["Topic"] == topic].sort_values("Timestamp")
        if len(tdf) == 0:
            continue
        words = topic_model.get_topic(topic) or []
        label = ", ".join([w for w, _ in words[:3]]) if words else str(topic)
        y = tdf["FrequencySmoothed"] if "FrequencySmoothed" in tdf else tdf["Frequency"]
        ax.plot(tdf["Timestamp"], y, marker="o", linewidth=2, label=f"Topic {topic}: {label}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    ax.set_title("Topic Evolution Over Time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "topic_evolution_line.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"已保存: {output_dir / 'topic_evolution_line.png'}")


def save_visualizations(topic_model, topics_over_time, df, output_dir, full_output=False):
    if topics_over_time is not None:
        if full_output:
            try:
                fig_time = topic_model.visualize_topics_over_time(
                    topics_over_time, top_n_topics=12, width=1200, height=600
                )
                fig_time.write_html(str(output_dir / "bertopic_topics_over_time.html"))
                log(f"已保存: {output_dir / 'bertopic_topics_over_time.html'}")
            except Exception as e:
                log(f"[WARN] 时间演化图失败: {e}")

    if full_output:
        try:
            fig_topics = topic_model.visualize_topics(width=1000, height=700)
            fig_topics.write_html(str(output_dir / "bertopic_topics_distribution.html"))
            log(f"已保存: {output_dir / 'bertopic_topics_distribution.html'}")
        except Exception as e:
            log(f"[WARN] 主题分布图失败: {e}")

        try:
            fig_heatmap = topic_model.visualize_heatmap(
                n_clusters=10, top_n_topics=20, width=1000, height=800
            )
            fig_heatmap.write_html(str(output_dir / "bertopic_heatmap.html"))
            log(f"已保存: {output_dir / 'bertopic_heatmap.html'}")
        except Exception as e:
            log(f"[WARN] 热力图失败: {e}")

        try:
            topics_per_class = topic_model.topics_per_class(
                df["content_tokenized"].tolist(),
                classes=df["ring_name"].tolist(),
                global_tuning=True,
            )
            fig_class = topic_model.visualize_topics_per_class(
                topics_per_class, top_n_topics=10, width=1200, height=700
            )
            fig_class.write_html(str(output_dir / "bertopic_topics_per_circle.html"))
            log(f"已保存: {output_dir / 'bertopic_topics_per_circle.html'}")
        except Exception as e:
            log(f"[WARN] 圈子分布图失败: {e}")

    if topics_over_time is not None:
        try:
            plot_topic_evolution(topic_model, topics_over_time, output_dir)
        except Exception as e:
            log(f"[WARN] 主题演化折线图失败: {e}")


def safe_topic_words(topic_model, topic_id, top_k=8):
    words = topic_model.get_topic(topic_id)
    if not words:
        return []
    out = []
    for w, _ in words:
        if is_generic_phrase(w):
            continue
        out.append(w)
        if len(out) >= top_k:
            break
    return out


def is_generic_phrase(text: str):
    if not isinstance(text, str):
        return True
    t = text.strip().lower()
    if not t:
        return True
    # 单字母/纯数字/符号
    if re.fullmatch(r"[\W_]+", t) or re.fullmatch(r"\d+", t):
        return True
    # 模板短语黑名单
    for g in Config.GENERIC_PHRASES:
        if g.replace(" ", "") in t.replace(" ", ""):
            return True
    return False


def trend_summary_for_topic(topic_id, topics_over_time):
    if topics_over_time is None:
        return "暂无时间趋势数据"
    tdf = topics_over_time[topics_over_time["Topic"] == topic_id].sort_values("Timestamp")
    if len(tdf) < 2:
        return "时间窗口不足，趋势待观察"

    y = tdf["FrequencySmoothed"].to_numpy() if "FrequencySmoothed" in tdf else tdf["Frequency"].to_numpy()
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    peak_idx = int(np.argmax(y))
    peak_time = str(pd.to_datetime(tdf.iloc[peak_idx]["Timestamp"]).date())
    if abs(slope) < 0.05 and abs(float(y[-1] - y[0])) <= 1.0:
        return f"整体平稳，峰值出现在 {peak_time}"
    if slope > 0:
        return f"整体上升，峰值出现在 {peak_time}"
    return f"整体回落，历史峰值出现在 {peak_time}"


def extract_topic_phrases(topic_docs_tokenized, core_keywords, top_k=5):
    if not topic_docs_tokenized:
        return []
    token_docs = [doc.split() for doc in topic_docs_tokenized if doc]
    if not token_docs:
        return []

    unigram_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()
    for tokens in token_docs:
        unigram_counts.update(tokens)
        for i in range(len(tokens) - 1):
            bg = (tokens[i], tokens[i + 1])
            if all(len(t) >= 2 for t in bg):
                bigram_counts[bg] += 1
        for i in range(len(tokens) - 2):
            tg = (tokens[i], tokens[i + 1], tokens[i + 2])
            if all(len(t) >= 2 for t in tg):
                trigram_counts[tg] += 1

    total_uni = sum(unigram_counts.values()) + Config.EPS
    total_bg = sum(bigram_counts.values()) + Config.EPS
    total_tg = sum(trigram_counts.values()) + Config.EPS
    core_set = set(core_keywords)
    scored = []

    for ng, cnt in bigram_counts.items():
        if cnt < Config.PHRASE_MIN_COUNT:
            continue
        p_ng = cnt / total_bg
        p_tokens = 1.0
        overlap = 0
        for t in ng:
            p_tokens *= unigram_counts[t] / total_uni + Config.EPS
            if t in core_set:
                overlap += 1
        pmi = math.log((p_ng + Config.EPS) / (p_tokens + Config.EPS))
        score = pmi * math.log(1 + cnt) + 0.25 * overlap
        scored.append((" ".join(ng), score))

    for ng, cnt in trigram_counts.items():
        if cnt < Config.PHRASE_MIN_COUNT:
            continue
        p_ng = cnt / total_tg
        p_tokens = 1.0
        overlap = 0
        for t in ng:
            p_tokens *= unigram_counts[t] / total_uni + Config.EPS
            if t in core_set:
                overlap += 1
        pmi = math.log((p_ng + Config.EPS) / (p_tokens + Config.EPS))
        score = pmi * math.log(1 + cnt) + 0.3 * overlap
        scored.append((" ".join(ng), score))

    scored.sort(key=lambda x: x[1], reverse=True)
    out = []
    seen = set()
    for phrase, _ in scored:
        if phrase in seen:
            continue
        if is_generic_phrase(phrase):
            continue
        seen.add(phrase)
        out.append(phrase)
        if len(out) >= top_k:
            break
    return out


def _text_jaccard_sim(text_a: str, text_b: str):
    sa = set(chinese_tokenize(text_a))
    sb = set(chinese_tokenize(text_b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / (len(sa | sb) + Config.EPS)


def select_representative_docs(docs, idxs, k=2):
    """从同一主题中选取去重后的代表文档。"""
    selected = []
    for idx in idxs:
        txt = docs[idx]
        is_dup = False
        for sidx in selected:
            if _text_jaccard_sim(txt, docs[sidx]) >= Config.DOC_DEDUP_SIM_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            selected.append(idx)
        if len(selected) >= k:
            break

    # 不足时再补（即使重复也补齐，避免空位）
    if len(selected) < k:
        for idx in idxs:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= k:
                break
    return selected[:k]


def build_topic_report(topic_model, df, topics, docs, topics_over_time, output_dir, full_output=False):
    topic_info = topic_model.get_topic_info()
    topic_to_indices = defaultdict(list)
    for i, t in enumerate(topics):
        topic_to_indices[int(t)].append(i)

    rows = []
    topic_profiles = {}
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue
        idxs = topic_to_indices.get(topic_id, [])
        core_keywords = safe_topic_words(topic_model, topic_id, Config.KEYWORD_TOP_K)
        topic_docs_tokenized = [df.loc[i, "content_tokenized"] for i in idxs[:400]]
        phrases = extract_topic_phrases(
            topic_docs_tokenized=topic_docs_tokenized,
            core_keywords=core_keywords,
            top_k=Config.PHRASE_TOP_K,
        )

        if phrases:
            topic_label = " / ".join(phrases[:2])
        elif core_keywords:
            topic_label = " / ".join(core_keywords[:2])
        else:
            topic_label = f"Topic {topic_id}"

        rep_docs = []
        rep_idxs = select_representative_docs(docs, idxs, k=2)
        for i in rep_idxs:
            c = docs[i]
            rep_docs.append(c[:180] + ("..." if len(c) > 180 else ""))
        while len(rep_docs) < 2:
            rep_docs.append("")

        rows.append(
            {
                "topic_id": topic_id,
                "topic_label": topic_label,
                "doc_count": len(idxs),
                "core_keywords": ", ".join(core_keywords),
                "representative_phrases": ", ".join(phrases),
                "representative_doc_1": rep_docs[0],
                "representative_doc_2": rep_docs[1],
                "trend_summary": trend_summary_for_topic(topic_id, topics_over_time),
            }
        )
        topic_profiles[topic_id] = {
            "label": topic_label,
            "keywords": core_keywords,
            "phrases": phrases,
            "has_representative_doc": bool(rep_docs[0].strip() or rep_docs[1].strip()),
        }

    report_df = pd.DataFrame(rows).sort_values("doc_count", ascending=False).reset_index(drop=True)
    report_path = output_dir / "topic_report.csv"
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    log(f"已保存: {report_path}")

    if full_output:
        audit_df = report_df.head(Config.MANUAL_AUDIT_SAMPLE_SIZE).copy()
        audit_df["readability_score"] = ""
        audit_df["naming_score"] = ""
        audit_df["noise_score"] = ""
        audit_df["reviewer"] = ""
        audit_df["review_date"] = ""
        audit_df["comments"] = ""
        audit_path = output_dir / "topic_manual_audit_template.csv"
        audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")
        log(f"已保存: {audit_path}")
    return report_df, topic_profiles


def jaccard_similarity(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / (len(a | b) + Config.EPS)


def build_topic_timeline(topic_model, topics_over_time, output_dir):
    result = {
        "generated_at": str(datetime.now()),
        "match_threshold": Config.TOPIC_MATCH_THRESHOLD,
        "split_threshold": Config.TOPIC_SPLIT_THRESHOLD,
        "events": [],
        "summary": {},
    }
    if topics_over_time is None or len(topics_over_time) == 0:
        result["summary"]["message"] = "topics_over_time 不可用，未生成轨迹事件"
        path = output_dir / "topic_timeline.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        log(f"已保存: {path}")
        return result

    df = topics_over_time.copy()
    df = df[df["Topic"] != -1].copy()
    if "FrequencyRaw" not in df.columns:
        df["FrequencyRaw"] = df["Frequency"]
    df = df[df["FrequencyRaw"] > 0].copy()

    bins = sorted(df["Timestamp"].unique())
    topic_kw = {
        int(t): set(safe_topic_words(topic_model, int(t), top_k=10))
        for t in df["Topic"].unique()
        if int(t) != -1
    }
    freq_by_bin = defaultdict(dict)
    for _, r in df.iterrows():
        freq_by_bin[r["Timestamp"]][int(r["Topic"])] = float(r["FrequencyRaw"])

    continuations = 0
    births = 0
    deaths = 0
    splits = 0

    for i in range(len(bins) - 1):
        t0, t1 = bins[i], bins[i + 1]
        prev_topics = sorted(freq_by_bin[t0].keys())
        curr_topics = sorted(freq_by_bin[t1].keys())
        sim = np.zeros((len(prev_topics), len(curr_topics)), dtype=float)

        for pi, pt in enumerate(prev_topics):
            for ci, ct in enumerate(curr_topics):
                sim[pi, ci] = jaccard_similarity(topic_kw.get(pt, set()), topic_kw.get(ct, set()))

        matched = []
        if sim.size > 0:
            r_idx, c_idx = linear_sum_assignment(1.0 - sim)
            for rr, cc in zip(r_idx, c_idx):
                score = float(sim[rr, cc])
                if score >= Config.TOPIC_MATCH_THRESHOLD:
                    matched.append((prev_topics[rr], curr_topics[cc], score))

        prev_matched = {x[0] for x in matched}
        curr_matched = {x[1] for x in matched}

        for src, dst, score in matched:
            result["events"].append(
                {
                    "type": "continuation",
                    "from_timestamp": str(pd.to_datetime(t0)),
                    "to_timestamp": str(pd.to_datetime(t1)),
                    "from_topic": int(src),
                    "to_topic": int(dst),
                    "similarity": score,
                }
            )
            continuations += 1

        for ct in curr_topics:
            if ct not in curr_matched:
                result["events"].append(
                    {"type": "newborn", "timestamp": str(pd.to_datetime(t1)), "topic": int(ct)}
                )
                births += 1
        for pt in prev_topics:
            if pt not in prev_matched:
                result["events"].append(
                    {"type": "disappeared", "timestamp": str(pd.to_datetime(t1)), "topic": int(pt)}
                )
                deaths += 1

        for pi, pt in enumerate(prev_topics):
            targets = []
            for ci, ct in enumerate(curr_topics):
                if sim[pi, ci] >= Config.TOPIC_SPLIT_THRESHOLD:
                    targets.append({"topic": int(ct), "similarity": float(sim[pi, ci])})
            if len(targets) > 1:
                result["events"].append(
                    {
                        "type": "split",
                        "from_timestamp": str(pd.to_datetime(t0)),
                        "to_timestamp": str(pd.to_datetime(t1)),
                        "from_topic": int(pt),
                        "to_topics": targets,
                    }
                )
                splits += 1

    result["summary"] = {
        "num_bins": int(len(bins)),
        "continuations": int(continuations),
        "newborns": int(births),
        "disappeared": int(deaths),
        "splits": int(splits),
    }
    path = output_dir / "topic_timeline.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"已保存: {path}")
    return result


def compute_coherence_metrics(topic_model, docs_tokenized, top_k=10):
    docs = [set(d.split()) for d in docs_tokenized if d]
    n_docs = len(docs)
    if n_docs == 0:
        return {"coherence_npmi": None, "coherence_c_v": None, "per_topic": {}}

    topic_ids = [int(t) for t in topic_model.get_topic_info()["Topic"].tolist() if int(t) != -1]
    topic_words = {tid: safe_topic_words(topic_model, tid, top_k=top_k) for tid in topic_ids}

    candidate_words = set()
    for ws in topic_words.values():
        candidate_words.update(ws)

    word_docs = defaultdict(set)
    for i, ws in enumerate(docs):
        for w in ws:
            if w in candidate_words:
                word_docs[w].add(i)

    def npmi(wi, wj):
        di = word_docs.get(wi, set())
        dj = word_docs.get(wj, set())
        pi = len(di) / (n_docs + Config.EPS)
        pj = len(dj) / (n_docs + Config.EPS)
        pij = len(di & dj) / (n_docs + Config.EPS)
        if pi <= 0 or pj <= 0 or pij <= 0:
            return -1.0
        pmi = math.log((pij + Config.EPS) / (pi * pj + Config.EPS))
        return pmi / (-math.log(pij + Config.EPS))

    npmi_all = []
    cv_all = []
    per_topic = {}
    for tid, words in topic_words.items():
        if len(words) < 2:
            per_topic[tid] = {"npmi": None, "c_v": None}
            continue

        pair_scores = [npmi(wi, wj) for wi, wj in combinations(words, 2)]
        topic_npmi = float(np.mean(pair_scores)) if pair_scores else None

        m = len(words)
        mat = np.zeros((m, m), dtype=float)
        for i in range(m):
            for j in range(m):
                mat[i, j] = 1.0 if i == j else npmi(words[i], words[j])
        cos_scores = []
        for i in range(m):
            for j in range(i + 1, m):
                a, b = mat[i], mat[j]
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na > 0 and nb > 0:
                    cos_scores.append(float(np.dot(a, b) / (na * nb + Config.EPS)))
        topic_cv = float(np.mean(cos_scores)) if cos_scores else None

        per_topic[tid] = {"npmi": topic_npmi, "c_v": topic_cv}
        if topic_npmi is not None:
            npmi_all.append(topic_npmi)
        if topic_cv is not None:
            cv_all.append(topic_cv)

    return {
        "coherence_npmi": float(np.mean(npmi_all)) if npmi_all else None,
        "coherence_c_v": float(np.mean(cv_all)) if cv_all else None,
        "per_topic": per_topic,
    }


def compute_topic_diversity(topic_model, top_k=10):
    topic_ids = [int(t) for t in topic_model.get_topic_info()["Topic"].tolist() if int(t) != -1]
    if not topic_ids:
        return None
    words = []
    for tid in topic_ids:
        words.extend(safe_topic_words(topic_model, tid, top_k))
    if not words:
        return None
    return float(len(set(words)) / (len(words) + Config.EPS))


def compute_stability_metrics(topics_over_time, topic_profiles):
    if topics_over_time is None or len(topics_over_time) == 0:
        return {
            "adjacent_window_consistency": None,
            "trend_correlation_raw_vs_smoothed": None,
            "burst_explainability_rate": None,
        }

    df = topics_over_time.copy()
    df = df[df["Topic"] != -1].copy()
    if "FrequencyRaw" not in df.columns:
        df["FrequencyRaw"] = df["Frequency"]
    if "FrequencySmoothed" not in df.columns:
        df["FrequencySmoothed"] = df["Frequency"]

    bins = sorted(df["Timestamp"].unique())
    topics = sorted(df["Topic"].unique())
    t2i = {t: i for i, t in enumerate(topics)}
    vecs = {}
    for b in bins:
        v = np.zeros(len(topics), dtype=float)
        sub = df[df["Timestamp"] == b]
        for _, r in sub.iterrows():
            v[t2i[r["Topic"]]] = float(r["FrequencyRaw"])
        vecs[b] = v

    adj = []
    for i in range(len(bins) - 1):
        v1, v2 = vecs[bins[i]], vecs[bins[i + 1]]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            adj.append(float(np.dot(v1, v2) / (n1 * n2 + Config.EPS)))
    adjacent_window_consistency = float(np.mean(adj)) if adj else None

    corr_list = []
    for tid in topics:
        tdf = df[df["Topic"] == tid].sort_values("Timestamp")
        if len(tdf) < 3:
            continue
        x = tdf["FrequencyRaw"].to_numpy(dtype=float)
        y = tdf["FrequencySmoothed"].to_numpy(dtype=float)
        if np.std(x) < Config.EPS or np.std(y) < Config.EPS:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if not np.isnan(corr):
            corr_list.append(corr)
    trend_corr = float(np.mean(corr_list)) if corr_list else None

    burst_total = 0
    burst_explainable = 0
    for tid in topics:
        tdf = df[df["Topic"] == tid].sort_values("Timestamp")
        vals = tdf["FrequencyRaw"].to_numpy(dtype=float)
        if len(vals) < 3:
            continue
        threshold = float(np.mean(vals) + np.std(vals))
        n_burst = int(np.sum(vals > threshold))
        if n_burst <= 0:
            continue
        burst_total += n_burst
        p = topic_profiles.get(int(tid), {})
        explainable = bool(p.get("phrases")) and bool(p.get("has_representative_doc"))
        if explainable:
            burst_explainable += n_burst
    burst_rate = float(burst_explainable / (burst_total + Config.EPS)) if burst_total else None

    return {
        "adjacent_window_consistency": adjacent_window_consistency,
        "trend_correlation_raw_vs_smoothed": trend_corr,
        "burst_explainability_rate": burst_rate,
    }


def build_metrics(topic_model, docs_tokenized, topics, topics_over_time, topic_profiles):
    coherence = compute_coherence_metrics(topic_model, docs_tokenized, top_k=10)
    diversity = compute_topic_diversity(topic_model, top_k=10)
    stability = compute_stability_metrics(topics_over_time, topic_profiles)

    topics_arr = np.array(topics)
    outlier_ratio = float(np.mean(topics_arr == -1)) if len(topics_arr) else None
    if outlier_ratio is not None and outlier_ratio > 0.35:
        log(f"[WARN] topic=-1 占比较高: {outlier_ratio:.2%}")

    return {
        "generated_at": str(datetime.now()),
        "quality": {
            "coherence_npmi": coherence["coherence_npmi"],
            "coherence_c_v": coherence["coherence_c_v"],
            "topic_diversity": diversity,
            "outlier_ratio_topic_minus_1": outlier_ratio,
        },
        "stability": stability,
        "per_topic_coherence": coherence["per_topic"],
    }


def save_metrics(metrics, output_dir):
    path = output_dir / "topic_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log(f"已保存: {path}")


def show_representative_docs(topic_model, df, topics, docs, top_k=5):
    topic_info = topic_model.get_topic_info()
    major_topics = (
        topic_info[topic_info["Topic"] != -1]
        .sort_values("Count", ascending=False)["Topic"]
        .head(top_k)
        .tolist()
    )
    topic_indices = defaultdict(list)
    for i, t in enumerate(topics):
        topic_indices[int(t)].append(i)
    for topic_id in major_topics:
        idxs = topic_indices.get(int(topic_id), [])
        kws = safe_topic_words(topic_model, int(topic_id), 5)
        log(f"\nTopic {topic_id} 关键词: {', '.join(kws)}")
        log(f"文档数: {len(idxs)}")
        for idx in select_representative_docs(docs, idxs, k=2):
            c = docs[idx]
            c = c[:150] + ("..." if len(c) > 150 else "")
            log(f"- 圈子: {df.loc[idx, 'ring_name']} | 时间: {df.loc[idx, 'pub_time']}")
            log(f"  内容: {c}")


def save_model(topic_model):
    path = Config.OUTPUT_DIR / "bertopic_model"
    topic_model.save(str(path))
    log(f"模型已保存: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="BERTopic Dynamic Topic Tracking (upgraded)")
    parser.add_argument("--pipeline", choices=["stable", "research"], default="stable")
    parser.add_argument("--time-granularity", choices=["month", "2week", "week"], default="month")
    parser.add_argument("--min-docs-per-bin", type=int, default=30)
    parser.add_argument("--smoothing", choices=["none", "rolling", "ewm"], default="rolling")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--export-metrics", action="store_true")
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="导出完整可视化和审阅模板；默认只导出核心结果",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_dir()

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"BERTopic Run - {datetime.now()}\n")
        f.write("=" * 60 + "\n")

    log("=" * 60)
    log("BERTopic Dynamic Topic Tracking (Upgraded)")
    log("=" * 60)
    log(
        "Args: "
        f"pipeline={args.pipeline}, time_granularity={args.time_granularity}, "
        f"min_docs_per_bin={args.min_docs_per_bin}, smoothing={args.smoothing}, "
        f"eval_only={args.eval_only}, export_metrics={args.export_metrics}, "
        f"full_output={args.full_output}"
    )

    try:
        df_raw = load_data()
        if len(df_raw) == 0:
            raise RuntimeError("输入数据为空")

        df_binned = build_time_bins(df_raw, args.time_granularity, args.min_docs_per_bin)
        df, docs, docs_tokenized, timestamps = preprocess_data(df_binned)
        if len(docs) == 0:
            raise RuntimeError("预处理后无有效文本")

        embedding_model = load_embedding_model()
        vectorizer_model = create_vectorizer()
        embeddings = generate_embeddings(embedding_model, docs)

        if args.eval_only:
            topic_model, topics, probs = load_model_for_eval(docs_tokenized, embeddings)
        else:
            topic_model, topics, probs = train_model(
                embedding_model, vectorizer_model, docs_tokenized, embeddings, args.pipeline
            )

        _ = probs
        show_topic_info(topic_model, rows=Config.TOPIC_INFO_PRINT_ROWS)

        topics_over_time = compute_topics_over_time(
            topic_model, docs_tokenized, timestamps, args.smoothing
        )
        save_visualizations(
            topic_model,
            topics_over_time,
            df,
            Config.OUTPUT_DIR,
            full_output=args.full_output,
        )

        report_df, topic_profiles = build_topic_report(
            topic_model,
            df,
            topics,
            docs,
            topics_over_time,
            Config.OUTPUT_DIR,
            full_output=args.full_output,
        )
        _ = report_df

        build_topic_timeline(topic_model, topics_over_time, Config.OUTPUT_DIR)
        metrics = build_metrics(topic_model, docs_tokenized, topics, topics_over_time, topic_profiles)
        save_metrics(metrics, Config.OUTPUT_DIR)

        if args.full_output:
            show_representative_docs(topic_model, df, topics, docs)

        if not args.eval_only:
            save_model(topic_model)

        log("=" * 60)
        log("实验完成")
        log("=" * 60)
    except Exception as e:
        log(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
