import os
import torch
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

def get_tokenizer(model_name=None):
    if model_name is None:
        model_name = os.getenv('SEN_EMB_MODEL_PATH')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    # 加载 sentence-transformers 模型
    model_path = os.getenv('SEN_EMB_MODEL_PATH')
    model = SentenceTransformer(model_path)

    texts = [
        # 天气
        "今天天气真好！",
        "今天天气不错。",
        # 电影
        "我喜欢这个电影。",
        "这个电影太棒了！",
        "这个电影不好看。",
        # 美食
        "这家餐厅很好吃。",
        "菜味道不错，推荐。",
        "太难吃了，不推荐。",
        # 运动
        "今天去跑步了，很累。",
        "运动完感觉真舒服。",
        # 工作
        "工作太多了，烦死了。",
        "终于下班了，开心！",
        # 科技
        "这个手机性能不错。",
        "电池续航太差了。",
        # 旅游
        "这里的风景真美。",
        "不想回家，太好玩了。",
    ]

    # 获取句向量
    embeddings = model.encode(texts, convert_to_tensor=True)  # [num_sentences, hidden_size]

    # 计算余弦相似度矩阵
    from sentence_transformers import util
    similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()  # [num_sentences, num_sentences]

    # 绘制热力图（纯 matplotlib）
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(similarity_matrix, cmap='YlOrRd', vmin=0, vmax=1)

    # 设置刻度和标签
    ax.set_xticks(np.arange(len(texts)))
    ax.set_yticks(np.arange(len(texts)))
    ax.set_xticklabels(texts)
    ax.set_yticklabels(texts)

    # 旋转 x 轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在每个格子里添加数值标注
    for i in range(len(texts)):
        for j in range(len(texts)):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity", rotation=270, labelpad=20)

    ax.set_title("Sentence Similarity Matrix (Sentence-Transformers)", fontsize=14)
    ax.set_xlabel("Sentences", fontsize=12)
    ax.set_ylabel("Sentences", fontsize=12)

    plt.tight_layout()
    plt.show()
