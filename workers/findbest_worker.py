# workers/findbest_worker.py
# LDA 決定最佳主題數的後台邏輯（從 script/lda_findBest_py.py 提取）

import os
import datetime
import threading

import numpy as np
import pandas as pd
import tomotopy as tp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import linalg

RM_POS = ["V_2", "DE", "SHI", "FW", "I", "T", "WHITESPACE"]

DEFAULT = {
    "seed": "777",
    "burnin": "500",
    "iteration": "2000",
    "thin": "100",
    "exclude_single_char": "0",
    "min_topic": "2",
    "max_topic": "10",
    "per_topic": "2",
}


def safe_int(text, default):
    try:
        v = int(text)
        return v if v > 0 else default
    except (ValueError, TypeError):
        return default


def safe_bool(text, default=True):
    if isinstance(text, bool):
        return text
    if text is None:
        return default
    s = str(text).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def make_filename(folder_path, filename_prefix, exclude_single_char):
    date = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    single_char_tag = "rm1char" if exclude_single_char else "keep1char"
    filename = f"{filename_prefix}_{single_char_tag}.xlsx"
    filename_date = f"{filename_prefix}_{single_char_tag}_{date}.xlsx"
    if os.path.exists(os.path.join(folder_path, filename)):
        return os.path.join(folder_path, filename_date)
    return os.path.join(folder_path, filename)


def process_files(input_file_paths):
    """讀取並前處理斷詞後的 xlsx，回傳各文件向量字串清單。"""
    ids, dfs, proc_dfs, vec_strs = [], [], [], []

    for path in input_file_paths:
        ids.append(os.path.basename(path))

        df = pd.read_excel(path)
        df.reset_index(inplace=True)
        df.rename(columns={"index": "raw_index"}, inplace=True)
        dfs.append(df)

        cols = ["raw_index", "sent_seq", "word", "pos", "sentsword_seq"]
        pre = df[cols].copy().dropna()
        pre = pre[~pre["pos"].str.endswith("CATEGORY")]
        pre = pre[~pre["pos"].isin(RM_POS)]
        pre.reset_index(drop=True, inplace=True)
        proc_dfs.append(pre)

        vec_strs.append(" ".join(pre["word"].values.tolist()))

    return ids, dfs, proc_dfs, vec_strs


def run_lda_full(
    word_lists, topic_nums, seed, burn_in, iteration, thin, exclude_single_char
):
    """對每個主題數 k 訓練 LDA 模型（collapsed Gibbs sampling）。"""
    if exclude_single_char:
        filtered = [[w for w in doc if len(w) >= 2] for doc in word_lists]
    else:
        filtered = word_lists
    total_tokens = sum(len(doc) for doc in filtered)

    models: dict = {}
    log_lik_samples: dict = {}

    for k in topic_nums:
        try:
            mdl = tp.LDAModel(k=k, seed=seed)
            for doc in filtered:
                if doc:
                    mdl.add_doc(doc)

            if len(mdl.docs) == 0:
                continue

            if burn_in > 0:
                mdl.train(burn_in, workers=1)

            post_iters = max(0, iteration)
            if thin > 0 and post_iters > 0:
                steps = max(1, post_iters // thin)
                ll_samples = []
                for _ in range(steps):
                    mdl.train(thin, workers=1)
                    ll_samples.append(mdl.ll_per_word * total_tokens)
            else:
                ll_samples = [mdl.ll_per_word * total_tokens]

            models[k] = mdl
            log_lik_samples[k] = ll_samples

        except Exception:
            pass

    return models, log_lik_samples, total_tokens


def _griffiths2004(log_lik_samples):
    result = {}
    for k, lls in log_lik_samples.items():
        x = np.array(lls, dtype=float)
        ll_med = np.median(x)
        result[k] = ll_med - np.log(np.mean(np.exp(-x + ll_med)))
    return result


def _caojuan2009(models):
    result = {}
    for k, mdl in models.items():
        phi = np.array([mdl.get_topic_word_dist(t) for t in range(k)])
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        sims = [
            np.dot(phi[i], phi[j]) / (np.linalg.norm(phi[i]) * np.linalg.norm(phi[j]))
            for i, j in pairs
        ]
        result[k] = np.sum(sims) / (k * (k - 1) / 2)
    return result


def _arun2010(models, word_lists):
    doc_lengths = np.array([len(doc) for doc in word_lists], dtype=float)
    eps = np.finfo(float).tiny
    result = {}
    for k, mdl in models.items():
        phi = np.array([mdl.get_topic_word_dist(t) for t in range(k)])
        _, sv, _ = linalg.svd(phi, full_matrices=False)
        cm1 = sv / sv.sum()

        gamma = np.array([doc.get_topic_dist() for doc in mdl.docs])
        cm2 = doc_lengths @ gamma
        cm2 = cm2 / np.max(np.abs(doc_lengths))
        cm2 = cm2 / cm2.sum()

        cm1 = np.clip(cm1, eps, None)
        cm2 = np.clip(cm2, eps, None)
        result[k] = float(
            np.sum(cm1 * np.log(cm1 / cm2)) + np.sum(cm2 * np.log(cm2 / cm1))
        )
    return result


def _deveaud2014(models):
    eps = np.finfo(float).tiny
    result = {}
    for k, mdl in models.items():
        phi = np.array([mdl.get_topic_word_dist(t) for t in range(k)])
        phi = np.where(phi == 0, eps, phi)
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        jsds = [
            0.5 * np.sum(phi[i] * np.log(phi[i] / phi[j]))
            + 0.5 * np.sum(phi[j] * np.log(phi[j] / phi[i]))
            for i, j in pairs
        ]
        result[k] = np.sum(jsds) / (k * (k - 1))
    return result


def _rescale_01(series):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - lo) / (hi - lo)


_MAXIMIZE = ["Griffiths2004", "Deveaud2014"]
_MINIMIZE = ["CaoJuan2009", "Arun2010"]
_MARKERS = {
    "Griffiths2004": "o",
    "CaoJuan2009": "s",
    "Arun2010": "^",
    "Deveaud2014": "D",
}


def save_metrics_plot(df_norm, save_path):
    """儲存雙面板指標折線圖。"""
    ks = df_norm["k"].tolist()
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle("Number of Topics Selection", fontsize=13)

    for panel_label, metrics, ax in [
        ("minimize", _MINIMIZE, axes[0]),
        ("maximize", _MAXIMIZE, axes[1]),
    ]:
        for metric in metrics:
            ax.plot(
                ks,
                df_norm[metric].tolist(),
                marker=_MARKERS[metric],
                label=metric,
                linewidth=1.5,
                markersize=7,
            )
        ax.set_ylabel(panel_label, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
        ax.grid(axis="x", color="grey", linewidth=0.7, alpha=0.6)
        ax.grid(axis="y", visible=False)
        ax.legend(fontsize=9, frameon=False)

    axes[-1].set_xlabel("number of topics", fontsize=10)
    axes[-1].set_xticks(ks)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_metrics_df(models, log_lik_samples, word_lists):
    """計算四個指標並建立含原始值與正規化值的 DataFrame。"""
    valid_ks = sorted(models.keys())
    g = _griffiths2004({k: log_lik_samples[k] for k in valid_ks})
    c = _caojuan2009({k: models[k] for k in valid_ks})
    a = _arun2010({k: models[k] for k in valid_ks}, word_lists)
    d = _deveaud2014({k: models[k] for k in valid_ks})

    rows = [
        {
            "k": k,
            "Griffiths2004": g[k],
            "CaoJuan2009": c[k],
            "Arun2010": a[k],
            "Deveaud2014": d[k],
        }
        for k in valid_ks
    ]
    df = pd.DataFrame(rows)

    for col in ["Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"]:
        df[f"Norm_{col}"] = _rescale_01(df[col])

    return df


def run_findbest(
    input_files,
    output_folder,
    seed,
    burn_in,
    iteration,
    thin,
    min_topic,
    max_topic,
    per_topic,
    exclude_single_char,
    on_append,
    on_done,
    on_error,
):
    """啟動背景執行緒執行 LDA 指標計算。"""

    def _worker():
        try:
            on_append(
                f"SEED: {seed}; Burn In: {burn_in}; Iteration: {iteration}; Thin: {thin}"
            )
            on_append(
                f"Topic Number From: {min_topic}; To: {max_topic}; By: {per_topic}"
            )
            on_append(f'排除單字詞：{"是" if exclude_single_char else "否"}')
            topic_nums = list(range(min_topic, max_topic + 1, per_topic))
            on_append(f"主題數清單: {topic_nums}")

            ids, _, _, vec_strs = process_files(input_files)
            word_lists = [text.split() for text in vec_strs]

            on_append("LDA 模型訓練中（依主題數數量可能需要較長時間）...")
            models, log_lik_samples, _ = run_lda_full(
                word_lists,
                topic_nums,
                seed,
                burn_in,
                iteration,
                thin,
                exclude_single_char,
            )

            if not models:
                on_error("計算失敗：無有效模型產生，請確認語料內容。")
                return

            on_append("計算四個評估指標...")
            result_df = build_metrics_df(models, log_lik_samples, word_lists)

            out_xlsx = make_filename(output_folder, "LDA_Metrics", exclude_single_char)
            result_df.to_excel(out_xlsx, index=False)
            on_append(f"Excel 已存至：{out_xlsx}")

            norm_cols = [
                "k",
                "Norm_Griffiths2004",
                "Norm_CaoJuan2009",
                "Norm_Arun2010",
                "Norm_Deveaud2014",
            ]
            df_plot = result_df[norm_cols].rename(
                columns={
                    "Norm_Griffiths2004": "Griffiths2004",
                    "Norm_CaoJuan2009": "CaoJuan2009",
                    "Norm_Arun2010": "Arun2010",
                    "Norm_Deveaud2014": "Deveaud2014",
                }
            )
            out_png = out_xlsx.replace(".xlsx", ".png")
            save_metrics_plot(df_plot, out_png)
            on_append(f"圖片已存至：{out_png}\n")

            on_done(out_png)

        except Exception as e:
            on_error(str(e))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
