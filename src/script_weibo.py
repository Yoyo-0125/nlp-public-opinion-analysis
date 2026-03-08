import argparse
from pathlib import Path

from dotenv import load_dotenv
import yaml

from data_crawler import WeiboPipelineCrawler
from topic_detector import DynamicTopicDetector, TopicDetectionConfig


def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def load_yaml_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"YAML 配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError("YAML 配置格式错误，顶层必须是对象（key-value）")
    return loaded


def resolve_params(args: argparse.Namespace, yaml_cfg: dict) -> dict:
    default_params = {
        "max_terms": 20,
        "max_pages": 3,
        "time_granularity": "week",
        "min_docs_per_bin": 40,
        "top_k": 15,
        "min_term_df": 2,
        "flow_top_n": 50,
        "console_top_k": 10,
        "console_recent_bins": 8,
    }
    params = dict(default_params)

    yaml_params = yaml_cfg.get("params", {})
    if isinstance(yaml_params, dict):
        for key in params:
            if key in yaml_params and yaml_params[key] is not None:
                params[key] = yaml_params[key]

    for key in params:
        value = getattr(args, key)
        if value is not None:
            params[key] = value

    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微博抓取 + 动态热点分析统一入口")
    parser.add_argument("--config", type=str, default="script_weibo.yaml", help="YAML 配置文件路径")
    parser.add_argument(
        "--skip-crawl",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否跳过抓取（覆盖 YAML）",
    )
    parser.add_argument("--weibo-dir", type=str, default=None, help="微博数据目录（读取 weibo_hot_pipeline_*.json）")
    parser.add_argument("--weibo-file", type=str, default=None, help="指定单个微博数据文件，优先于 --weibo-dir")
    parser.add_argument("--output-dir", type=str, default=None, help="分析结果输出目录")
    parser.add_argument(
        "--with-visuals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否生成图表与HTML报告（覆盖 YAML）",
    )

    advanced = parser.add_argument_group("advanced")
    advanced.add_argument("--max-terms", type=int, default=None, help="覆盖 YAML：抓取阶段处理前 N 个热榜词条")
    advanced.add_argument("--max-pages", type=int, default=None, help="覆盖 YAML：抓取阶段每个词条抓取页数")
    advanced.add_argument("--time-granularity", choices=["month", "2week", "week", "day"], default=None, help="覆盖 YAML：时间粒度")
    advanced.add_argument("--min-docs-per-bin", type=int, default=None, help="覆盖 YAML：每个时间窗最少文档数")
    advanced.add_argument("--top-k", type=int, default=None, help="覆盖 YAML：每个时间窗输出热点数")
    advanced.add_argument("--min-term-df", type=int, default=None, help="覆盖 YAML：最小词频阈值")
    advanced.add_argument("--flow-top-n", type=int, default=None, help="覆盖 YAML：热点流动跟踪词数")
    advanced.add_argument("--console-top-k", type=int, default=None, help="覆盖 YAML：终端每窗显示数量")
    advanced.add_argument("--console-recent-bins", type=int, default=None, help="覆盖 YAML：终端显示最近时间窗数")
    advanced.add_argument("--extra-stopwords", type=str, default=None, help="覆盖 YAML：额外停用词，逗号分隔")
    advanced.add_argument("--stopwords-file", type=str, default=None, help="覆盖 YAML：外部停用词文件路径（每行一个词）")
    advanced.add_argument("--priority-terms", type=str, default=None, help="覆盖 YAML：优先关键词，逗号分隔")
    advanced.add_argument("--priority-boost", type=float, default=None, help="覆盖 YAML：优先关键词分数放大倍率")
    advanced.add_argument(
        "--console-no-doc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="终端是否显示代表文本（覆盖 YAML）",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    yaml_cfg = load_yaml_config(Path(args.config))

    project_root = Path(__file__).parent.parent
    params = resolve_params(args, yaml_cfg)

    skip_crawl = (
        args.skip_crawl
        if args.skip_crawl is not None
        else bool(yaml_cfg.get("skip_crawl", False))
    )
    with_visuals = (
        args.with_visuals
        if args.with_visuals is not None
        else bool(yaml_cfg.get("with_visuals", False))
    )
    extra_stop_words = yaml_cfg.get("extra_stopwords", [])
    if not isinstance(extra_stop_words, list):
        extra_stop_words = []
    stop_words_file = yaml_cfg.get("stopwords_file", "")
    priority_terms = yaml_cfg.get("priority_terms", [])
    if not isinstance(priority_terms, list):
        priority_terms = []
    priority_boost = float(yaml_cfg.get("priority_boost", 1.0))
    if args.extra_stopwords is not None:
        extra_stop_words = parse_csv_list(args.extra_stopwords)
    if args.stopwords_file is not None:
        stop_words_file = args.stopwords_file
    if args.priority_terms is not None:
        priority_terms = parse_csv_list(args.priority_terms)
    if args.priority_boost is not None:
        priority_boost = float(args.priority_boost)

    console_no_doc = (
        args.console_no_doc
        if args.console_no_doc is not None
        else bool(yaml_cfg.get("console_no_doc", False))
    )
    weibo_dir_str = args.weibo_dir if args.weibo_dir is not None else str(yaml_cfg.get("weibo_dir", "data"))
    weibo_file_str = args.weibo_file if args.weibo_file is not None else str(yaml_cfg.get("weibo_file", ""))
    output_dir_str = args.output_dir if args.output_dir is not None else str(yaml_cfg.get("output_dir", "outputs"))

    weibo_dir = Path(weibo_dir_str)

    print(f"[Config] YAML: {args.config}")

    if not skip_crawl:
        crawler = WeiboPipelineCrawler()
        crawl_output = crawler.run(
            max_terms=params["max_terms"],
            max_pages=params["max_pages"],
            output_dir=weibo_dir,
        )
        print(f"[Pipeline] 抓取完成: {crawl_output}")

    data_source = Path(weibo_file_str) if weibo_file_str.strip() else weibo_dir

    config = TopicDetectionConfig(
        project_root=project_root,
        data_file=data_source,
        circles_file=project_root / "data" / "zhihu_ai_circles.json",
        output_dir=Path(output_dir_str),
        use_weibo=True,
        time_granularity=params["time_granularity"],
        min_docs_per_bin=params["min_docs_per_bin"],
        top_k=params["top_k"],
        min_term_df=params["min_term_df"],
        flow_top_n=params["flow_top_n"],
        console_top_k=params["console_top_k"],
        console_recent_bins=params["console_recent_bins"],
        console_no_doc=console_no_doc,
        with_visuals=with_visuals,
        extra_stop_words=extra_stop_words,
        stop_words_file=str(stop_words_file).strip() or None,
        priority_terms=priority_terms,
        priority_boost=priority_boost,
    )
    detector = DynamicTopicDetector(config)
    result = detector.run()
    print(
        "[Pipeline] 分析完成: "
        f"time_bins={result['valid_time_bins']}, "
        f"hotspots={len(result['hotspots'])}, flows={len(result['flows'])}"
    )


if __name__ == "__main__":
    main()
