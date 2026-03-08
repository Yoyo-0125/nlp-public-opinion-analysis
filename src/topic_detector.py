"""
Dynamic hotspot analysis without BERTopic.

This script extracts keyword/phrase hotspots over time by burst scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

# Suppress third-party deprecation noise from jieba/setuptools.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import jieba
import jieba.posseg as pseg
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Paths:
    project_root: Path
    data_file: Path
    circles_file: Path
    output_dir: Path


@dataclass
class TopicDetectionConfig:
    project_root: Path
    data_file: Path
    circles_file: Path
    output_dir: Path
    use_weibo: bool = False
    time_granularity: str = "week"
    min_docs_per_bin: int = 30
    top_k: int = 20
    min_term_df: int = 2
    flow_top_n: int = 30
    console_top_k: int = 8
    console_recent_bins: int = 5
    console_no_doc: bool = False
    with_visuals: bool = False
    extra_stop_words: List[str] | None = None
    stop_words_file: str | None = None
    priority_terms: List[str] | None = None
    priority_boost: float = 1.0


class KeywordPhraseExtractor:
    """Extract cleaned tokens and ngram phrases from Chinese text."""

    def __init__(self, stop_words: set[str] | None = None):
        self.stop_words = stop_words or set()
        self.valid_pos = {"n", "nr", "ns", "nt", "nz", "vn", "eng", "x", "nw", "nx"}
        self.verb_whitelist = {
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
            "支持",
            "停售",
        }
        self.generic_phrases = {
            "关键 结论",
            "论文 详解",
            "详解 核心",
            "核心 要点",
            "一句话",
            "推荐 清单",
            "欢迎 交流",
            "以上 内容",
        }
        self.strip_punct = string.punctuation + "，。！？；：、（）【】《》“”‘’—…·「」『』"

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"#[^#\s]+#?", "", text)
        text = text.replace("\u200b", " ").replace("\ufeff", " ").replace("\u2060", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _normalize_term(self, term: str) -> str:
        if not isinstance(term, str):
            return ""
        t = unicodedata.normalize("NFKC", term)
        t = t.replace("\u200b", "").replace("\ufeff", "").replace("\u2060", "")
        t = t.strip()
        t = t.strip(self.strip_punct)
        t = re.sub(r"\s+", " ", t)
        return t

    def _is_valid_term(self, term: str) -> bool:
        if not term:
            return False
        # must contain at least one Chinese/Latin/digit char
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", term):
            return False
        # remove pure punctuation/symbol tokens and emoji-like artifacts
        if re.fullmatch(r"[\W_]+", term):
            return False
        # avoid artifacts like "✔", "️", "⃣", isolated punctuation fragments
        if any(ch in term for ch in {"✔", "️", "⃣", "★", "☆", "•"}):
            return False
        return True

    def _compose_phrase(self, tokens: List[str]) -> str:
        # Pure Chinese phrases are concatenated to reduce duplicate variants like "后浪 研究所"
        all_cjk = all(re.fullmatch(r"[\u4e00-\u9fff0-9]+", t or "") for t in tokens)
        if all_cjk:
            return "".join(tokens)
        return " ".join(tokens)

    def _is_generic_phrase(self, phrase: str) -> bool:
        if not phrase:
            return True
        low = phrase.lower()
        normalized = low.replace(" ", "")
        for g in self.generic_phrases:
            if g.replace(" ", "") in normalized:
                return True
        return False

    def tokenize(self, text: str) -> List[str]:
        text = self.clean_text(text)
        if len(text) < 2:
            return []

        out: List[str] = []
        for word, pos in pseg.lcut(text):
            word = self._normalize_term(word)
            if not word:
                continue
            if word in self.stop_words:
                continue
            if word.isdigit():
                continue
            if not self._is_valid_term(word):
                continue

            has_latin = bool(re.search(r"[A-Za-z]", word))
            has_cjk = bool(re.search(r"[\u4e00-\u9fff]", word))
            if has_latin and len(word) < 2:
                continue
            if has_cjk and len(word) < 2:
                continue

            if pos == "v" and word not in self.verb_whitelist:
                continue
            if pos in self.valid_pos or pos == "v" or has_latin:
                out.append(word)
        # keep order but deduplicate
        return list(dict.fromkeys(out))

    def extract_terms(self, text: str, ngram_range: tuple[int, int] = (1, 3)) -> List[str]:
        tokens = self.tokenize(text)
        if not tokens:
            return []

        n_min, n_max = ngram_range
        terms: List[str] = []
        for n in range(n_min, n_max + 1):
            if n == 1:
                terms.extend([t for t in tokens if self._is_valid_term(t)])
                continue
            for i in range(0, len(tokens) - n + 1):
                phrase = self._normalize_term(self._compose_phrase(tokens[i : i + n]))
                if not self._is_valid_term(phrase):
                    continue
                if self._is_generic_phrase(phrase):
                    continue
                terms.append(phrase)
        return list(dict.fromkeys(terms))


class BurstHotspotModel:
    """
    Burst scoring model for dynamic hotspots.

    Score combines:
    - current frequency in bin
    - lift against historical mean
    - z-like burst score
    - inverse document frequency across all bins
    """

    def __init__(
        self,
        min_term_df: int = 2,
        priority_terms: List[str] | None = None,
        priority_boost: float = 1.0,
    ):
        self.min_term_df = min_term_df
        self.bin_term_df: List[Counter[str]] = []
        self.bin_docs: List[pd.DataFrame] = []
        self.bin_labels: List[str] = []
        self.global_df: Counter[str] = Counter()
        self.priority_terms = [t.strip().lower() for t in (priority_terms or []) if isinstance(t, str) and t.strip()]
        self.priority_boost = max(1.0, float(priority_boost))

    def _priority_multiplier(self, term: str) -> float:
        if not self.priority_terms:
            return 1.0
        low = str(term).lower()
        for p in self.priority_terms:
            if p and p in low:
                return self.priority_boost
        return 1.0

    def fit(self, grouped_docs: Sequence[tuple[str, pd.DataFrame]]) -> None:
        self.bin_term_df = []
        self.bin_docs = []
        self.bin_labels = []
        self.global_df = Counter()

        for label, gdf in grouped_docs:
            term_df: Counter[str] = Counter()
            for terms in gdf["terms"].tolist():
                # use document frequency within bin
                for t in set(terms):
                    term_df[t] += 1
            self.bin_term_df.append(term_df)
            self.bin_docs.append(gdf)
            self.bin_labels.append(label)
            self.global_df.update(term_df.keys())

    def _idf(self, term: str) -> float:
        n_bins = len(self.bin_term_df)
        df = self.global_df.get(term, 0)
        return math.log((n_bins + 1.0) / (df + 1.0)) + 1.0

    def _history_stats(self, term: str, idx: int) -> tuple[float, float]:
        if idx <= 0:
            return 0.0, 0.0
        history_vals = [self.bin_term_df[i].get(term, 0) for i in range(idx)]
        return float(np.mean(history_vals)), float(np.std(history_vals))

    def score_bin(self, idx: int, top_k: int = 20) -> List[dict]:
        curr = self.bin_term_df[idx]
        scored: List[dict] = []
        for term, freq in curr.items():
            if freq < self.min_term_df:
                continue
            mean_h, std_h = self._history_stats(term, idx)
            lift = (freq + 1.0) / (mean_h + 1.0)
            z = (freq - mean_h) / (std_h + 1.0)
            burst = max(0.0, z)
            score = (
                0.7 * math.log1p(freq)
                + 0.9 * math.log1p(lift)
                + 0.8 * burst
            ) * self._idf(term)
            score *= self._priority_multiplier(term)

            scored.append(
                {
                    "term": term,
                    "score": float(score),
                    "freq": int(freq),
                    "history_mean": float(mean_h),
                    "lift": float(lift),
                    "burst_z": float(z),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def build_hotspots(self, top_k: int = 20, rep_docs: int = 2) -> List[dict]:
        results: List[dict] = []
        for idx, label in enumerate(self.bin_labels):
            hotspots = self.score_bin(idx, top_k=top_k)
            gdf = self.bin_docs[idx]
            for h in hotspots:
                term = h["term"]
                matched = gdf[gdf["terms"].apply(lambda ts: term in set(ts))]
                if len(matched) > 0:
                    matched = matched.sort_values(["likes"], ascending=False)
                    reps = matched["content"].head(rep_docs).tolist()
                else:
                    reps = []
                h["time_bin"] = label
                h["representative_docs"] = reps
                results.append(h)
        return results

    def term_time_series(self, term: str) -> List[dict]:
        series = []
        for i, label in enumerate(self.bin_labels):
            series.append(
                {
                    "time_bin": label,
                    "freq": int(self.bin_term_df[i].get(term, 0)),
                }
            )
        return series


def default_stop_words() -> set[str]:
    return {
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
        "对于",
        "因为",
        "所以",
        "但是",
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
        "然后",
        "最后",
        "比如",
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
        "gpt",
        "llm",
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
        "分析",
        "内容",
        "文章",
    }


def load_stop_words_from_file(file_path: str | None) -> set[str]:
    if not file_path:
        return set()
    path = Path(file_path)
    if not path.exists():
        print(f"[Warning] stopwords file not found: {path}")
        return set()
    words: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                words.add(w)
    return words


def load_data(paths: Paths, use_weibo: bool) -> pd.DataFrame:
    if use_weibo:
        data_path = paths.data_file
        if data_path.is_dir():
            files = sorted(data_path.glob("weibo_hot_pipeline_*.json"))
        else:
            files = [data_path]

        posts = []
        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                payload = json.load(f)
            posts.extend(payload.get("all_posts", []))

        df = pd.DataFrame(posts)
        df["content"] = df.get("text", "")
        df["likes"] = pd.to_numeric(df.get("likes", 0), errors="coerce").fillna(0).astype(int)
        # Use mixed parsing to avoid format inference warnings on heterogeneous timestamps.
        try:
            df["pub_time"] = pd.to_datetime(
                df.get("timestamp", ""),
                errors="coerce",
                utc=True,
                format="mixed",
            )
        except TypeError:
            # Fallback for older pandas versions.
            df["pub_time"] = pd.to_datetime(df.get("timestamp", ""), errors="coerce", utc=True)
        if isinstance(df["pub_time"].dtype, pd.DatetimeTZDtype):
            df["pub_time"] = (
                df["pub_time"]
                .dt.tz_convert("Asia/Shanghai")
                .dt.tz_localize(None)
            )
        df = df.dropna(subset=["pub_time", "content"]).reset_index(drop=True)
        df["ring_name"] = "weibo"
        return df

    with open(paths.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    if paths.circles_file.exists() and "ring_id" in df.columns:
        with open(paths.circles_file, "r", encoding="utf-8") as f:
            circles_data = json.load(f)
        ring_to_name = {c["ring_id"]: c["name"] for c in circles_data}
        df["ring_name"] = df["ring_id"].map(ring_to_name).fillna(df["ring_id"])
    else:
        df["ring_name"] = ""

    if "likes" not in df.columns:
        df["likes"] = 0
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)

    df["pub_time"] = pd.to_datetime(df["pub_time"], errors="coerce")
    df = df.dropna(subset=["pub_time", "content"]).reset_index(drop=True)
    return df


def assign_time_bin(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    out = df.copy()
    if granularity == "month":
        out["time_bin"] = out["pub_time"].dt.to_period("M").astype(str)
    elif granularity == "week":
        out["time_bin"] = out["pub_time"].dt.to_period("W-MON").astype(str)
    elif granularity == "2week":
        base = out["pub_time"].min().floor("D")
        delta = ((out["pub_time"] - base).dt.days // 14).astype(int)
        out["time_bin"] = (
            (base + pd.to_timedelta(delta * 14, unit="D")).dt.strftime("%Y-%m-%d")
            if hasattr(base, "dt")
            else (base + pd.to_timedelta(delta * 14, unit="D")).astype(str)
        )
    else:
        if granularity == "day":
            out["time_bin"] = out["pub_time"].dt.strftime("%Y-%m-%d")
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")
    return out


def prepare_terms(df: pd.DataFrame, extractor: KeywordPhraseExtractor) -> pd.DataFrame:
    out = df.copy()
    out["content"] = out["content"].astype(str)
    out["terms"] = out["content"].apply(extractor.extract_terms)
    out = out[out["terms"].apply(len) > 0].reset_index(drop=True)
    return out


def group_bins(df: pd.DataFrame, min_docs_per_bin: int) -> List[tuple[str, pd.DataFrame]]:
    grouped = []
    for key, gdf in df.groupby("time_bin", sort=True):
        if len(gdf) >= min_docs_per_bin:
            grouped.append((str(key), gdf.copy()))
    return grouped


def _granularity_fallback_order(preferred: str) -> List[str]:
    order = ["month", "2week", "week", "day"]
    if preferred in order:
        start = order.index(preferred)
        return order[start:]
    return ["week", "day"]


def build_grouped_bins_with_fallback(
    df: pd.DataFrame,
    preferred_granularity: str,
    min_docs_per_bin: int,
) -> tuple[List[tuple[str, pd.DataFrame]], str, int]:
    for granularity in _granularity_fallback_order(preferred_granularity):
        binned = assign_time_bin(df, granularity)
        # Try user threshold first, then gradually relax to 1.
        for current_min_docs in [min_docs_per_bin, max(1, min_docs_per_bin // 2), 10, 5, 1]:
            grouped = group_bins(binned, min_docs_per_bin=current_min_docs)
            if len(grouped) >= 2:
                return grouped, granularity, current_min_docs
    # Last fallback: keep preferred binning and allow single-bin output for static hotspots.
    binned = assign_time_bin(df, preferred_granularity if preferred_granularity in {"month", "2week", "week", "day"} else "week")
    grouped = group_bins(binned, min_docs_per_bin=1)
    return grouped, preferred_granularity, 1


def save_outputs(hotspots: List[dict], paths: Paths, top_k: int) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = paths.output_dir / "keyword_hotspots.json"
    csv_path = paths.output_dir / "keyword_hotspots.csv"
    summary_path = paths.output_dir / "keyword_hotspots_summary.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(hotspots, f, ensure_ascii=False, indent=2)

    rows = []
    for h in hotspots:
        rows.append(
            {
                "time_bin": h["time_bin"],
                "term": h["term"],
                "score": h["score"],
                "freq": h["freq"],
                "history_mean": h["history_mean"],
                "lift": h["lift"],
                "burst_z": h["burst_z"],
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    # concise summary for terminal users
    summary_lines = []
    summary_lines.append("Keyword Hotspots Summary")
    summary_lines.append("=" * 40)
    by_bin: Dict[str, List[dict]] = defaultdict(list)
    for h in hotspots:
        by_bin[h["time_bin"]].append(h)
    for tb in sorted(by_bin.keys()):
        summary_lines.append(f"\n[{tb}] top {min(top_k, len(by_bin[tb]))}")
        for item in by_bin[tb][:top_k]:
            summary_lines.append(
                f"- {item['term']} | score={item['score']:.3f} | freq={item['freq']} | lift={item['lift']:.2f}"
            )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")


def _trend_label(freqs: List[int]) -> str:
    if not freqs:
        return "unknown"
    non_zero_idx = [i for i, f in enumerate(freqs) if f > 0]
    if len(non_zero_idx) <= 1:
        return "new_or_sparse"
    x = np.arange(len(freqs), dtype=float)
    y = np.array(freqs, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(freqs) >= 2 else 0.0
    delta = y[-1] - y[0]
    if slope > 0.05 and delta > 0:
        return "rising"
    if slope < -0.05 and delta < 0:
        return "falling"
    return "stable"


def build_hotspot_flows(
    hotspots: List[dict], model: BurstHotspotModel, top_n_terms: int = 30
) -> List[dict]:
    """Build time-flow trajectories for top hotspot terms."""
    term_scores: Dict[str, float] = defaultdict(float)
    term_peak: Dict[str, float] = defaultdict(float)
    for h in hotspots:
        t = h["term"]
        s = float(h["score"])
        term_scores[t] += s
        if s > term_peak[t]:
            term_peak[t] = s

    ranked_terms = sorted(
        term_scores.keys(),
        key=lambda t: (term_scores[t], term_peak[t]),
        reverse=True,
    )[:top_n_terms]

    flows: List[dict] = []
    for term in ranked_terms:
        series = model.term_time_series(term)
        freqs = [p["freq"] for p in series]
        if max(freqs) <= 0:
            continue
        peak_idx = int(np.argmax(freqs))
        flows.append(
            {
                "term": term,
                "trend": _trend_label(freqs),
                "peak_freq": int(max(freqs)),
                "peak_time_bin": series[peak_idx]["time_bin"],
                "total_freq": int(sum(freqs)),
                "time_series": series,
            }
        )
    return flows


def save_hotspot_flows(flows: List[dict], paths: Paths) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = paths.output_dir / "keyword_hotspot_flows.json"
    csv_path = paths.output_dir / "keyword_hotspot_flows.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(flows, f, ensure_ascii=False, indent=2)

    rows = []
    for flow in flows:
        for p in flow["time_series"]:
            rows.append(
                {
                    "term": flow["term"],
                    "trend": flow["trend"],
                    "peak_time_bin": flow["peak_time_bin"],
                    "peak_freq": flow["peak_freq"],
                    "total_freq": flow["total_freq"],
                    "time_bin": p["time_bin"],
                    "freq": p["freq"],
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


def _sanitize_for_display(term: str) -> str:
    if not isinstance(term, str):
        return ""
    t = re.sub(r"\s+", " ", term).strip()
    # reduce punctuation-only artifacts
    if re.fullmatch(r"[\W_]+", t):
        return ""
    return t


def generate_visual_reports(
    hotspots: List[dict],
    flows: List[dict],
    paths: Paths,
    top_line_terms: int = 8,
    top_heat_terms: int = 15,
) -> None:
    """Generate visual outputs: line chart, heatmap, and HTML dashboard."""
    if not hotspots:
        return
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    # Filter display-ready flows
    clean_flows = [f for f in flows if _sanitize_for_display(f.get("term", ""))]
    if not clean_flows:
        clean_flows = flows

    # Build score ranking from hotspots
    term_total_score: Dict[str, float] = defaultdict(float)
    for h in hotspots:
        term = _sanitize_for_display(h.get("term", ""))
        if not term:
            continue
        term_total_score[term] += float(h.get("score", 0.0))

    ranked_terms = [t for t, _ in sorted(term_total_score.items(), key=lambda x: x[1], reverse=True)]

    # Map flow by term
    flow_map = {f["term"]: f for f in clean_flows}
    usable_terms = [t for t in ranked_terms if t in flow_map]
    if not usable_terms:
        usable_terms = [f["term"] for f in clean_flows]

    # line chart
    line_terms = usable_terms[:top_line_terms]
    line_png = paths.output_dir / "keyword_hotspot_flow_lines.png"
    if line_terms:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(14, 7))
        for term in line_terms:
            series = flow_map[term]["time_series"]
            x = [p["time_bin"] for p in series]
            y = [p["freq"] for p in series]
            ax.plot(x, y, marker="o", linewidth=2, label=term)
        ax.set_title("Hotspot Flow Over Time (Top Terms)")
        ax.set_xlabel("Time Bin")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(line_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {line_png}")

    # heatmap (terms x bins)
    heat_terms = usable_terms[:top_heat_terms]
    heat_png = paths.output_dir / "keyword_hotspot_flow_heatmap.png"
    if heat_terms:
        bins = flow_map[heat_terms[0]]["time_series"]
        x_labels = [p["time_bin"] for p in bins]
        mat = []
        for term in heat_terms:
            series = flow_map[term]["time_series"]
            mat.append([p["freq"] for p in series])
        mat_arr = np.array(mat, dtype=float)

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(mat_arr, aspect="auto", interpolation="nearest")
        ax.set_title("Hotspot Flow Heatmap")
        ax.set_xlabel("Time Bin")
        ax.set_ylabel("Term")
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(heat_terms)))
        ax.set_yticklabels(heat_terms, fontsize=9)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Frequency")
        plt.tight_layout()
        plt.savefig(heat_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {heat_png}")

    # HTML dashboard
    html_path = paths.output_dir / "keyword_hotspot_dashboard.html"
    hot_df = pd.DataFrame(
        [
            {
                "time_bin": h.get("time_bin"),
                "term": _sanitize_for_display(h.get("term", "")),
                "score": round(float(h.get("score", 0.0)), 3),
                "freq": int(h.get("freq", 0)),
                "lift": round(float(h.get("lift", 0.0)), 2),
                "burst_z": round(float(h.get("burst_z", 0.0)), 2),
            }
            for h in hotspots
            if _sanitize_for_display(h.get("term", ""))
        ]
    )

    if len(hot_df) == 0:
        hot_df = pd.DataFrame(
            [{"time_bin": "", "term": "", "score": 0.0, "freq": 0, "lift": 0.0, "burst_z": 0.0}]
        )

    top_global = hot_df.sort_values("score", ascending=False).head(30)

    flow_df_rows = []
    for f in clean_flows:
        flow_df_rows.append(
            {
                "term": f.get("term", ""),
                "trend": f.get("trend", ""),
                "peak_time_bin": f.get("peak_time_bin", ""),
                "peak_freq": f.get("peak_freq", 0),
                "total_freq": f.get("total_freq", 0),
            }
        )
    flow_df = pd.DataFrame(flow_df_rows).sort_values("total_freq", ascending=False).head(30)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Keyword Hotspot Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color:#222; }}
    h1, h2 {{ margin: 8px 0; }}
    .meta {{ color:#555; margin-bottom:16px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; }}
    th {{ background: #f5f5f5; text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin: 8px 0 20px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
  </style>
</head>
<body>
  <h1>Keyword Hotspot Dashboard</h1>
  <div class="meta">Generated from keyword-only dynamic hotspot pipeline.</div>

  <h2>Flow Line Chart</h2>
  <img src="{line_png.name}" alt="flow lines"/>

  <h2>Flow Heatmap</h2>
  <img src="{heat_png.name}" alt="flow heatmap"/>

  <h2>Top Global Hotspots</h2>
  {top_global.to_html(index=False, escape=False)}

  <h2>Top Hotspot Flows</h2>
  {flow_df.to_html(index=False, escape=False)}
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {html_path}")


def _short_text(text: str, max_len: int = 90) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def print_hotspot_console_summary(
    hotspots: List[dict],
    top_k: int = 8,
    recent_bins: int = 5,
    show_docs: bool = True,
) -> None:
    """Print useful hotspot summary directly in terminal."""
    if not hotspots:
        print("\nNo hotspots to display.")
        return

    by_bin: Dict[str, List[dict]] = defaultdict(list)
    for h in hotspots:
        by_bin[h["time_bin"]].append(h)

    ordered_bins = sorted(by_bin.keys())
    if recent_bins > 0:
        ordered_bins = ordered_bins[-recent_bins:]

    print("\n" + "=" * 72)
    print("Hotspot Summary (Console)")
    print("=" * 72)
    for tb in ordered_bins:
        items = sorted(by_bin[tb], key=lambda x: x["score"], reverse=True)
        head = items[:top_k]
        print(f"\n[{tb}] top {len(head)} hotspots")
        for i, item in enumerate(head, start=1):
            term = item["term"]
            score = item["score"]
            freq = item["freq"]
            lift = item["lift"]
            burst = item["burst_z"]
            print(
                f"{i:>2}. {term} | score={score:.3f} | freq={freq} | "
                f"lift={lift:.2f} | burst_z={burst:.2f}"
            )
            if show_docs:
                reps = item.get("representative_docs", [])
                if reps:
                    print(f"    example: {_short_text(reps[0], max_len=100)}")

    global_top = sorted(hotspots, key=lambda x: x["score"], reverse=True)[:top_k]
    print("\nGlobal Top Hotspots")
    for i, item in enumerate(global_top, start=1):
        print(
            f"{i:>2}. [{item['time_bin']}] {item['term']} | "
            f"score={item['score']:.3f} | lift={item['lift']:.2f}"
        )


class DynamicTopicDetector:
    """High-level runner for keyword-based dynamic hotspot detection."""

    def __init__(self, config: TopicDetectionConfig):
        self.config = config

    def _build_paths(self) -> Paths:
        return Paths(
            project_root=self.config.project_root,
            data_file=self.config.data_file,
            circles_file=self.config.circles_file,
            output_dir=self.config.output_dir,
        )

    def run(self) -> dict:
        paths = self._build_paths()

        print("Loading data...")
        df = load_data(paths, use_weibo=self.config.use_weibo)
        print(f"Loaded documents: {len(df)}")

        stop_words = default_stop_words()
        stop_words.update(load_stop_words_from_file(self.config.stop_words_file))
        if self.config.extra_stop_words:
            stop_words.update(
                {str(w).strip() for w in self.config.extra_stop_words if str(w).strip()}
            )
        extractor = KeywordPhraseExtractor(stop_words=stop_words)
        df = prepare_terms(df, extractor)
        print(f"Valid documents after term extraction: {len(df)}")

        grouped, used_granularity, used_min_docs = build_grouped_bins_with_fallback(
            df,
            preferred_granularity=self.config.time_granularity,
            min_docs_per_bin=self.config.min_docs_per_bin,
        )
        if (
            used_granularity != self.config.time_granularity
            or used_min_docs != self.config.min_docs_per_bin
        ):
            print(
                f"[AutoFallback] granularity: {self.config.time_granularity} -> {used_granularity}, "
                f"min_docs_per_bin: {self.config.min_docs_per_bin} -> {used_min_docs}"
            )
        if len(grouped) < 1:
            raise RuntimeError("No valid time bins after fallback. Please check input data quality.")
        if len(grouped) < 2:
            print("[Warning] Only 1 valid time bin. Dynamic flow will be limited; output downgraded to static hotspots.")
        print(f"Valid time bins: {len(grouped)}")

        model = BurstHotspotModel(
            min_term_df=self.config.min_term_df,
            priority_terms=self.config.priority_terms,
            priority_boost=self.config.priority_boost,
        )
        model.fit(grouped)
        hotspots = model.build_hotspots(top_k=self.config.top_k, rep_docs=2)
        print(f"Extracted hotspots: {len(hotspots)}")

        print_hotspot_console_summary(
            hotspots,
            top_k=self.config.console_top_k,
            recent_bins=self.config.console_recent_bins,
            show_docs=not self.config.console_no_doc,
        )

        save_outputs(hotspots, paths, top_k=self.config.top_k)

        flows = build_hotspot_flows(hotspots, model, top_n_terms=self.config.flow_top_n)
        save_hotspot_flows(flows, paths)
        print(f"Built hotspot flows: {len(flows)}")

        if self.config.with_visuals:
            generate_visual_reports(hotspots, flows, paths)

        return {
            "hotspots": hotspots,
            "flows": flows,
            "used_granularity": used_granularity,
            "used_min_docs_per_bin": used_min_docs,
            "valid_time_bins": len(grouped),
        }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Keyword-only dynamic hotspot analysis")
    parser.add_argument(
        "--data-file",
        type=str,
        default=str(project_root / "data" / "zhihu_ring_data_20260225_senti.json"),
    )
    parser.add_argument(
        "--circles-file",
        type=str,
        default=str(project_root / "data" / "zhihu_ai_circles.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "outputs"),
    )
    parser.add_argument("--time-granularity", choices=["month", "2week", "week", "day"], default="week")
    parser.add_argument("--min-docs-per-bin", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-term-df", type=int, default=2)
    parser.add_argument(
        "--flow-top-n",
        type=int,
        default=30,
        help="构建时间流动轨迹的热点词数量（按综合热点得分）",
    )
    parser.add_argument(
        "--use-weibo",
        action="store_true",
        help="从微博 pipeline 结果读取而不是知乎圈子数据",
    )
    parser.add_argument(
        "--console-top-k",
        type=int,
        default=8,
        help="终端每个时间窗展示的热点数量",
    )
    parser.add_argument(
        "--console-recent-bins",
        type=int,
        default=5,
        help="终端仅展示最近 N 个时间窗（<=0 表示全部）",
    )
    parser.add_argument(
        "--console-no-doc",
        action="store_true",
        help="终端摘要不展示代表文本",
    )
    parser.add_argument(
        "--weibo-dir",
        type=str,
        default="data",
        help="微博 pipeline 结果目录（合并所有 weibo_hot_pipeline_*.json）",
    )
    parser.add_argument(
        "--with-visuals",
        action="store_true",
        help="生成图片与HTML可视化报告（默认关闭以提升速度与稳定性）",
    )
    parser.add_argument(
        "--extra-stopwords",
        type=str,
        default="",
        help="额外停用词，逗号分隔",
    )
    parser.add_argument(
        "--priority-terms",
        type=str,
        default="",
        help="优先关键词，逗号分隔（如 openclaw,moltbot,glm）",
    )
    parser.add_argument(
        "--priority-boost",
        type=float,
        default=1.0,
        help="优先关键词分数放大倍率（>=1.0）",
    )
    parser.add_argument(
        "--stopwords-file",
        type=str,
        default="",
        help="外部停用词文件路径（每行一个词）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra_stop_words = [x.strip() for x in args.extra_stopwords.split(",") if x.strip()]
    priority_terms = [x.strip() for x in args.priority_terms.split(",") if x.strip()]
    config = TopicDetectionConfig(
        project_root=Path(__file__).parent.parent,
        data_file=Path(args.weibo_dir) if args.use_weibo else Path(args.data_file),
        circles_file=Path(args.circles_file),
        output_dir=Path(args.output_dir),
        use_weibo=args.use_weibo,
        time_granularity=args.time_granularity,
        min_docs_per_bin=args.min_docs_per_bin,
        top_k=args.top_k,
        min_term_df=args.min_term_df,
        flow_top_n=args.flow_top_n,
        console_top_k=args.console_top_k,
        console_recent_bins=args.console_recent_bins,
        console_no_doc=args.console_no_doc,
        with_visuals=args.with_visuals,
        extra_stop_words=extra_stop_words,
        stop_words_file=args.stopwords_file or None,
        priority_terms=priority_terms,
        priority_boost=args.priority_boost,
    )
    detector = DynamicTopicDetector(config)
    detector.run()


if __name__ == "__main__":
    main()
