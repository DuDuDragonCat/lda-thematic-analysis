# workers/findbest_worker.py
# LDA 決定最佳主題數的後台邏輯（從 script/lda_findBest_py.py 提取）

import os
import datetime
import threading

import mpmath
import numpy as np
import pandas as pd
import tomotopy as tp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import linalg
from scipy.special import gammaln

RM_POS = ["V_2", "DE", "SHI", "FW", "I", "T", "WHITESPACE"]

DEFAULT = {
    "seed": "777",
    "burnin": "500",
    "iteration": "1000",
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


def _log_p_w_given_z(mdl):
    """計算 log P(w|z)，對齊 R topicmodels 的 @logLiks。

    為什麼不能直接用 tomotopy 的 ll_per_word：
      R topicmodels（Gibbs）的 @logLiks 記錄的是 Griffiths & Steyvers (2004)
      式 (2) 的 log P(w|z)——只含 topic-word 那一項；ldatuning 的
      Griffiths2004 指標（調和平均估計）在數學上也只對這個量成立。
      而 tomotopy 的 ll_per_word 是完整 joint log P(w,z|α,η)，多了
      document-topic 先驗項 log P(z|α)。該項的量級隨 k 與 α（=50/k）平移，
      會把小 k 的 likelihood 灌高，導致 Griffiths 曲線形狀被扭曲
      （例如 k=2 反而變成最大值）。

    Griffiths & Steyvers (2004) 式 (2)：
      log P(w|z) = k·[lgamma(V·η) − V·lgamma(η)]
                   + Σ_t [ Σ_w lgamma(n_tw + η) − lgamma(n_t + V·η) ]

    tomotopy 未直接提供 topic-word 計數 n_tw，但它回傳的 φ 就是
    φ_tw = (n_tw + η) / (n_t + V·η)，因此 n_tw + η = φ_tw · (n_t + V·η)，
    可從 φ 與 get_count_by_topics() 的 n_t 無損還原公式所需的每一項。
    """
    k = mdl.k
    eta = float(mdl.eta)
    V = len(mdl.used_vocabs)
    n_t = np.array(mdl.get_count_by_topics(), dtype=float)

    ll = k * (gammaln(V * eta) - V * gammaln(eta))
    for t in range(k):
        phi = np.array(mdl.get_topic_word_dist(t), dtype=float)
        denom = n_t[t] + V * eta
        ll += float(np.sum(gammaln(phi * denom)) - gammaln(denom))
    return ll


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
            # 對齊 R topicmodels Gibbs 預設：alpha=50/k、delta(eta)=0.1，
            # 且訓練期間超參數固定（optim_interval=0 關閉自動優化）。
            mdl = tp.LDAModel(k=k, alpha=50.0 / k, eta=0.1, seed=seed)
            mdl.optim_interval = 0
            for doc in filtered:
                if doc:
                    mdl.add_doc(doc)

            if len(mdl.docs) == 0:
                continue

            if burn_in > 0:
                mdl.train(burn_in, workers=1)

            post_iters = max(0, iteration)
            # 對齊 R topicmodels：每 thin 個迭代記錄一次 log P(w|z)
            # （相當於 topicmodels 的 keep 參數與 @logLiks），
            # 不用 ll_per_word（joint likelihood，見 _log_p_w_given_z 說明）。
            if thin > 0 and post_iters > 0:
                steps = max(1, post_iters // thin)
                ll_samples = []
                for _ in range(steps):
                    mdl.train(thin, workers=1)
                    ll_samples.append(_log_p_w_given_z(mdl))
            else:
                ll_samples = [_log_p_w_given_z(mdl)]

            models[k] = mdl
            log_lik_samples[k] = ll_samples

        except Exception:
            pass

    return models, log_lik_samples, total_tokens


# 對應 R ldatuning 套件 Rmpfr::mpfr(x, prec=2000L) 的位元精度
_MPFR_PRECISION_BITS = 2000


def _griffiths2004(log_lik_samples):
    """調和平均數估計法（Griffiths & Steyvers, 2004）。

    語料庫總對數概似量級大（total_tokens 很大時，樣本間差距動輒上千），
    直接用 float64 算 exp(-x + ll_med) 會溢位成 inf，導致結果變成 -inf。
    這裡改用 mpmath 任意精度運算（等同 R ldatuning 套件用 Rmpfr 的做法），
    在高精度下計算 exp/mean/log，最後才轉回一般 float 回傳。
    """
    result = {}
    with mpmath.workprec(_MPFR_PRECISION_BITS):
        for k, lls in log_lik_samples.items():
            x = np.array(lls, dtype=float)
            ll_med = float(np.median(x))
            terms = [mpmath.exp(mpmath.mpf(ll_med - xi)) for xi in x]
            mean_val = mpmath.fsum(terms) / len(terms)
            result[k] = float(ll_med - mpmath.log(mean_val))
    return result


# ── 替代做法：log-sum-exp ────────────────────────────────────────────────
# 用 log-sum-exp 技巧取代任意精度運算，同樣能避免 exp() 溢位，且不需要
# mpmath。數學上與上面的 mpmath 版本等價，但精度略低（仍是 float64）。
# 若想改用這個版本：把上面的 _griffiths2004 註解掉，並取消下面的註解。
#
# from scipy.special import logsumexp
#
# def _griffiths2004(log_lik_samples):
#     result = {}
#     for k, lls in log_lik_samples.items():
#         x = np.array(lls, dtype=float)
#         ll_med = np.median(x)
#         y = ll_med - x
#         log_mean_exp = logsumexp(y) - np.log(len(y))
#         result[k] = ll_med - log_mean_exp
#     return result


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


def _arun2010(models):
    """對齊 R ldatuning 套件的 Arun2010 原始碼。

    注意：Arun et al. (2010) 論文描述的是「兩個機率分布間的對稱 KL
    divergence」，但 ldatuning 的 R 原始碼實際上「不做」正規化——
      cm1 <- m1.svd$d                    # 原始奇異值，未除以總和
      cm2 <- (len %*% gamma) / norm(len,"m")  # 只除以最大文件長度，未除以總和
    R 版數值落在 ~200 的量級、且曲線平緩遞減，正是來自這種未正規化的
    向量。若把 cm1、cm2 各自正規化成總和為 1 的分布（數學上較符合論文），
    數值會縮到個位數且曲線形狀翻轉，就對不上 R 的結果。
    這裡照抄 ldatuning 的行為以求一致。

    文件長度取自 mdl.docs（訓練時實際使用的 token 數），對應 R 版
    slam::row_sums(dtm)；不可用過濾前的原始詞列表，否則排除單字詞時
    長度會偏大，且空文件被跳過時會與 gamma 的列錯位。
    """
    eps = np.finfo(float).tiny
    result = {}
    for k, mdl in models.items():
        phi = np.array([mdl.get_topic_word_dist(t) for t in range(k)])
        _, sv, _ = linalg.svd(phi, full_matrices=False)
        cm1 = sv

        doc_lengths = np.array([len(doc.words) for doc in mdl.docs], dtype=float)
        gamma = np.array([doc.get_topic_dist() for doc in mdl.docs])
        cm2 = doc_lengths @ gamma
        cm2 = cm2 / np.max(np.abs(doc_lengths))

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


def build_metrics_df(models, log_lik_samples):
    """計算四個指標並建立含原始值與正規化值的 DataFrame。"""
    valid_ks = sorted(models.keys())
    g = _griffiths2004({k: log_lik_samples[k] for k in valid_ks})
    c = _caojuan2009({k: models[k] for k in valid_ks})
    a = _arun2010({k: models[k] for k in valid_ks})
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
            result_df = build_metrics_df(models, log_lik_samples)

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
