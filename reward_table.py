"""reward_table.py

把 RL 环境里每个 step 都要重算一次的 GNN 前向，改成"每个 (样本, 动作) 只算一次"的查表。

唯一的强约束
------------
查表结果必须与原始逐次调用 ``evaluate_gnn`` 的结果**逐位相同**。

为此本文件刻意做了三件事，改动时请勿破坏：

1. 不重写计算逻辑。表中的每一条都是通过调用调用方传进来的 ``reward_fn``
   （即 env_train.evaluate_gnn 本身）得到的，保证走的是同一条代码路径。
2. 不做批处理。``dgl.batch`` 会把 GEMM 的 M 维从 1 变成 B，cuBLAS 会选不同的
   kernel，浮点累加顺序随之改变，产生 ~1e-6 的相对差异。单图逐条前向则与原
   实现完全一致。
3. loss 以 float32 存放。原始 ``loss.item()`` 是一个 fp32 标量被提升成 python
   float，写回 fp32 数组是无损往返。
"""

import hashlib
import json
import os

import numpy as np

# 只要改了"reward 是怎么算的"（evaluate_gnn 的逻辑、loss_func 的类型/超参、
# build_graph 的语义），就必须把这里 +1，否则旧表会被静默复用。
# v2: 指纹改为对数据内容做哈希（v1 用的是 label 文件 mtime）。
TABLE_SCHEMA_VERSION = "v2"


def model_fingerprint(paths, salt=""):
    """用"路径 + 大小 + mtime"给一组模型文件生成指纹。

    表缓存只有在 4 个 reward 模型都没变的情况下才可复用。任何 pickle 的变化
    （重新预训练、换 seed、换 fold）都会让指纹变化，从而强制重算。

    salt 用来把"不体现在模型文件里、但影响表内容"的配置也拼进来
    （典型是 loss_func 的类型与超参）。
    """
    h = hashlib.sha1()
    h.update(f"schema={TABLE_SCHEMA_VERSION}|salt={salt}".encode())
    for p in paths:
        try:
            st = os.stat(p)
            h.update(f"{os.path.abspath(p)}|{st.st_size}|{int(st.st_mtime)}".encode())
        except OSError:
            # 文件不存在时让指纹必然失效，避免误用旧表
            h.update(f"missing|{p}".encode())
    return h.hexdigest()[:16]


def loss_salt(loss_func):
    """把 loss 配置压成字符串并进指纹。

    reward 表里存的是**已经算好的 loss**，所以 loss_func 一旦变了，旧表就失效。
    这里把类型与常见超参抽出来，避免"改了 label_smoothing 却复用了旧表"。
    """
    attrs = []
    for name in ("label_smoothing", "reduction", "ignore_index"):
        val = getattr(loss_func, name, None)
        if val is not None:
            attrs.append(f"{name}={val}")
    return f"{type(loss_func).__name__}({','.join(attrs)})"


def data_fingerprint(dataset):
    """用样本**内容**（features_name 的每个字段 + label）生成指纹。

    为什么不用文件 mtime：data_process.py 每次重跑都会重写 npy，mtime 必变，
    缓存就永远命不中；反过来，mtime 没变也不代表内容没变（拷贝、回滚、覆盖写)。
    只查 label 文件则更糟——activity/duration 变了而标签和样本数没变时，旧表会被
    静默复用，结果错得没有任何提示。

    所以直接对真正参与构图和前向的数值做哈希。'_tid' / '_obs' 是 env_train 加的
    运行时字段，不参与计算，也不进指纹。
    """
    h = hashlib.sha1()
    h.update(f"schema={TABLE_SCHEMA_VERSION}".encode())
    for d in dataset:
        for k in sorted(k for k in d if not k.startswith('_')):
            h.update(k.encode())
            v = d[k]
            try:
                arr = np.asarray(v)
            except (ValueError, TypeError):
                # 长度不齐的嵌套序列在新版 numpy 上会抛 ValueError（旧版只是 warning）。
                # 指纹这里没必要卡死，退成 repr 一样能保证"内容变则指纹变"。
                h.update(repr(v).encode())
                continue
            if arr.dtype == object:
                # 非常规结构（如嵌套数组）退回 repr；Python 3 的 float repr 可精确往返
                h.update(repr(v).encode())
            else:
                h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:16]


def table_fingerprint(model_paths, loss_func, dataset):
    """一张表的完整指纹：reward 模型文件 + loss 配置 + 该 split 的数据内容。"""
    return model_fingerprint(
        model_paths,
        salt=f"{loss_salt(loss_func)}|data={data_fingerprint(dataset)}",
    )


class RewardTable:
    """(样本数, 动作数) 的 reward 查表。

    三个数组分别对应原始 evaluate_gnn 的返回三元组：
        loss    : float32, 单样本交叉熵（= loss_func(logits, label) 的值）
        correct : bool,    预测是否命中
        pred    : int32,   logits.argmax(1)
    """

    __slots__ = ("loss", "correct", "pred", "fingerprint")

    def __init__(self, loss, correct, pred, fingerprint=None):
        self.loss = loss
        self.correct = correct
        self.pred = pred
        self.fingerprint = fingerprint

    # ---- 基本属性 ----
    @property
    def n_samples(self):
        return int(self.loss.shape[0])

    @property
    def n_actions(self):
        return int(self.loss.shape[1])

    def __len__(self):
        return self.n_samples

    # ---- 取数：返回 numpy 标量，保持与原 torch tensor 一样的 .item() 接口 ----
    def get(self, idx, action):
        return (
            self.loss[idx, action],       # np.float32，有 .item()
            int(self.correct[idx, action]),
            int(self.pred[idx, action]),
        )

    def row(self, idx):
        """一次取出一行的三个数组切片，供批量统计使用。"""
        return self.loss[idx], self.correct[idx], self.pred[idx]

    # ---- 落盘 ----
    def save(self, path, meta=None):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        info = dict(meta or {})
        # 用 update 而不是 dict(meta, fingerprint=...)：后者在 meta 自带同名键时
        # 会抛 "got multiple values for keyword argument"，属于纯自找的崩溃点。
        info.update(fingerprint=self.fingerprint,
                    n_samples=self.n_samples, n_actions=self.n_actions)
        payload = dict(
            loss=self.loss,
            correct=self.correct,
            pred=self.pred,
            meta=json.dumps(info),
        )
        # np.savez 不希望自动追加 .npz 时传文件句柄
        with open(path, "wb") as f:
            np.savez_compressed(f, **payload)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            info = json.loads(str(z["meta"]))
            return cls(z["loss"], z["correct"], z["pred"], info.get("fingerprint"))


def build_reward_table(data, reward_fn, build_fn, n_actions, verbose=True, log_every=2000):
    """逐 (样本, 动作) 计算 reward 表。

    参数
    ----
    data        : list[dict]，每个元素含 'activity' / 'label' 等特征
    reward_fn   : callable(graph, attr_label, action) -> (loss, correct, pred)
                  —— 传入 env_train.evaluate_gnn，不要传别的实现
    build_fn    : callable(attr_label, action) -> dgl graph（已在目标 device 上）
    n_actions   : 动作数（4）
    """
    n = len(data)
    loss = np.zeros((n, n_actions), dtype=np.float32)
    correct = np.zeros((n, n_actions), dtype=bool)
    pred = np.zeros((n, n_actions), dtype=np.int32)

    for i, d in enumerate(data):
        for a in range(n_actions):
            graph = build_fn(d, a)
            l, c, p = reward_fn(graph, d, a)
            # .item() 得到的 python float 由 fp32 无损提升而来，写回 fp32 无损
            loss[i, a] = l.item()
            correct[i, a] = bool(c)
            pred[i, a] = int(p)
            # 及时释放，避免 4N 张图同时驻留显存
            del graph
        if verbose and (i + 1) % log_every == 0:
            print(f"    reward table: {i + 1}/{n} 样本已完成", flush=True)

    return RewardTable(loss, correct, pred)


def build_or_load(data, reward_fn, build_fn, n_actions, cache_path=None,
                  fingerprint=None, verbose=True):
    """带磁盘缓存的表构建。

    命中条件：文件存在 + 指纹一致 + 形状一致。任一不满足则重算并覆盖。
    """
    if cache_path and os.path.exists(cache_path):
        try:
            tbl = RewardTable.load(cache_path)
            if (tbl.n_samples == len(data) and tbl.n_actions == n_actions
                    and (fingerprint is None or tbl.fingerprint == fingerprint)):
                if verbose:
                    print(f"    命中缓存 {cache_path}（{tbl.n_samples}×{tbl.n_actions}）")
                return tbl
            if verbose:
                print(f"    缓存 {cache_path} 与当前数据/模型不一致，重算")
        except Exception as exc:  # 缓存损坏就当没有
            if verbose:
                print(f"    缓存读取失败（{exc}），重算")

    tbl = build_reward_table(data, reward_fn, build_fn, n_actions, verbose=verbose)
    tbl.fingerprint = fingerprint
    if cache_path:
        tbl.save(cache_path, meta={"n": len(data)})
        if verbose:
            print(f"    已写入 {cache_path}")
    return tbl
