from node2vec import Node2Vec
import networkx as nx
import os

# 폴더 미리 생성
os.makedirs("../datasets/STRING_dataset/go-terms/emb/", exist_ok=True)

# 그래프 로딩
G = nx.read_edgelist("../datasets/STRING_dataset/go-terms/graph/go-terms.edgelist", create_using=nx.DiGraph())

# Node2Vec 모델 정의
model = Node2Vec(
    G,
    dimensions=64,
    walk_length=30,
    num_walks=200,
    p=1, q=1,
    workers=4
)

# 학습 및 저장
wv_model = model.fit(window=10, min_count=1)
wv_model.wv.save_word2vec_format("../datasets/STRING_dataset/go-terms/emb/go-terms-64.emd")
