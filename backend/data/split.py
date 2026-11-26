"""数据划分（防泄漏）—— 固定种子、类别分层、可选按批次隔离。

两种模式：
  - "stratified"：按类别分层随机划分（当前 10 类分布均衡），固定种子可复现。
  - "by_batch"  ：整批次隔离——某些批次整体作测试集，更能反映对「重新采集数据」
                  的泛化能力；验证集从训练批次内按分层carve出。

划分保证三集索引互不相交；行走轨迹应整条作为一个 group，不拆散（用 groups 参数）。

自检运行：python -m backend.data.split
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from backend import config


def _batch_sort_key(b: str):
    """按批次名里的数字排序（"r2" < "r10"），而不是字符串字典序（"r10" < "r2"）。"""
    m = re.search(r"\d+", b)
    return int(m.group()) if m else b


def _stratified_indices(labels, idx_pool, ratios, rng):
    """在 idx_pool 上按类别分层划分为 (train, val, test) 三份索引。"""
    train, val, test = [], [], []
    for c in np.unique(labels[idx_pool]):
        idx = idx_pool[labels[idx_pool] == c]
        rng.shuffle(idx)
        n = len(idx)
        a = int(ratios[0] * n)
        b = int((ratios[0] + ratios[1]) * n)
        train.append(idx[:a]); val.append(idx[a:b]); test.append(idx[b:])
    return (np.concatenate(train) if train else np.array([], int),
            np.concatenate(val) if val else np.array([], int),
            np.concatenate(test) if test else np.array([], int))


def make_split(labels, batches=None, groups=None,
               ratios=(0.70, 0.15, 0.15), seed: int = config.RANDOM_SEED,
               mode: str = "stratified",
               test_batches=None, val_batches=None) -> dict:
    """返回 {'train','val','test'} 三个索引数组。

    test_batches/val_batches：仅 by_batch 模式下生效，显式指定整批次分配给
    test/val（其余批次归 train）。传了其中任一个就走显式分配，忽略 ratios；
    都不传则保留原先按比例自动切的行为（向后兼容）。
    """
    labels = np.asarray(labels)
    N = len(labels)
    rng = np.random.default_rng(seed)
    all_idx = np.arange(N)

    if mode == "stratified":
        if groups is not None:
            # 组不拆散：整组分配到某一集合，再在组层面近似满足比例
            ug = np.unique(groups)
            rng.shuffle(ug)
            na = int(ratios[0] * len(ug))
            nb = int((ratios[0] + ratios[1]) * len(ug))
            g2set = {g: ("train" if i < na else "val" if i < nb else "test")
                     for i, g in enumerate(ug)}
            sset = np.array([g2set[g] for g in groups])
            tr, va, te = all_idx[sset == "train"], all_idx[sset == "val"], all_idx[sset == "test"]
        else:
            tr, va, te = _stratified_indices(labels, all_idx, ratios, rng)

    elif mode == "by_batch":
        if batches is None:
            raise ValueError("by_batch 模式需要提供 batches")
        batches = np.asarray(batches)
        ub = sorted(set(batches.tolist()), key=_batch_sort_key)
        if len(ub) < 2:
            raise ValueError(f"by_batch 需要至少 2 个批次，当前只有 {ub}")

        if test_batches is not None or val_batches is not None:
            test_b, val_b = set(test_batches or []), set(val_batches or [])
            unknown = (test_b | val_b) - set(ub)
            if unknown:
                raise ValueError(f"test_batches/val_batches 里有不存在的批次: {unknown}（现有批次: {ub}）")
            te = all_idx[np.isin(batches, list(test_b))]
            va = all_idx[np.isin(batches, list(val_b))]
            tr = all_idx[~np.isin(batches, list(test_b | val_b))]
        else:
            n_test_b = max(1, round(ratios[2] * len(ub)))
            test_b = set(ub[-n_test_b:])
            te = all_idx[np.isin(batches, list(test_b))]
            rest = all_idx[~np.isin(batches, list(test_b))]
            # 剩余批次内按分层carve验证集
            val_frac = ratios[1] / (ratios[0] + ratios[1])
            tr2, va2, _ = _stratified_indices(labels, rest, (1 - val_frac, val_frac, 0.0), rng)
            tr, va = tr2, va2
    else:
        raise ValueError(f"未知划分模式: {mode}")

    split = {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}
    _assert_disjoint(split, N)
    return split


def _assert_disjoint(split: dict, N: int) -> None:
    s_tr, s_va, s_te = (set(split[k].tolist()) for k in ("train", "val", "test"))
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "划分存在重叠，数据泄漏！"


def make_grouped_batch_folds(batches, n_splits: int = 5) -> list[dict]:
    """生成按采集批次隔离的循环交叉验证折。

    每个批次恰好作为测试集一次；紧随测试块后的一个批次作为验证集，其余作为训练集。
    返回的是批次名而非样本索引，供窗口构造后再统一切分。
    """
    unique = sorted(set(np.asarray(batches).tolist()), key=_batch_sort_key)
    if n_splits < 2 or len(unique) < n_splits + 1:
        raise ValueError(f"需要至少 n_splits+1 个批次，当前 batches={unique}, n_splits={n_splits}")
    blocks = [list(v) for v in np.array_split(np.asarray(unique), n_splits)]
    folds = []
    for i, test_batches in enumerate(blocks):
        last_position = max(unique.index(str(v)) for v in test_batches)
        val_batch = unique[(last_position + 1) % len(unique)]
        test_batches = [str(v) for v in test_batches]
        if val_batch in test_batches:
            raise RuntimeError("无法生成与测试批次分离的验证批次")
        train_batches = [v for v in unique if v not in set(test_batches + [val_batch])]
        folds.append({
            "fold": i + 1,
            "train_batches": train_batches,
            "validation_batches": [val_batch],
            "test_batches": test_batches,
        })
    tested = [b for fold in folds for b in fold["test_batches"]]
    if sorted(tested, key=_batch_sort_key) != unique:
        raise RuntimeError("交叉验证测试折没有恰好覆盖全部批次一次")
    return folds


def save_split(split: dict, path: Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: split[k].tolist() for k in ("train", "val", "test")}
    payload["_meta"] = meta or {}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    from backend.data.load import load_static

    d = load_static()
    labels, batches = d["label"], d["batch"]

    sp = make_split(labels, batches, mode="stratified")
    print(f"[分层] train/val/test = {len(sp['train'])}/{len(sp['val'])}/{len(sp['test'])}")
    print(f"       训练集覆盖类别数 = {len(np.unique(labels[sp['train']]))}/{config.NUM_CLASSES}")
    assert len(np.unique(labels[sp["train"]])) == config.NUM_CLASSES, "有类别未进入训练集"

    sp2 = make_split(labels, batches, mode="by_batch")
    tr_b = sorted(set(batches[sp2["train"]].tolist()))
    te_b = sorted(set(batches[sp2["test"]].tolist()))
    print(f"[批次隔离] train/val/test = {len(sp2['train'])}/{len(sp2['val'])}/{len(sp2['test'])}")
    print(f"           训练批次={tr_b}  测试批次={te_b}  (应无交集)")
    assert not (set(tr_b) & set(te_b)), "训练/测试批次有交集，泄漏！"

    print("划分自检通过：三集不相交、类别全覆盖、批次隔离成立。")
