# workers/tagger_worker.py
# 載入 CKIP 模型並執行斷詞的後台邏輯（從 script/tagger_files.py 提取）

import os
import threading

import pandas as pd

WS_FOLDER  = 'bert-base-chinese-ws'
POS_FOLDER = 'bert-base-chinese-pos'
DEVICE     = -1   # -1 = CPU


def load_model_worker(model_dir: str, on_append, on_done, on_error):
    """
    背景執行緒：從本地資料夾載入 CKIP 模型。

    Callbacks
    ---------
    on_append(msg)           — log 輸出
    on_done((ws, pos))       — 模型載入完成，回傳 (ws, pos) tuple
    on_error(msg)            — 發生錯誤
    """
    try:
        from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

        ws_path  = os.path.join(model_dir, WS_FOLDER)
        pos_path = os.path.join(model_dir, POS_FOLDER)

        for path, label in [(ws_path, WS_FOLDER), (pos_path, POS_FOLDER)]:
            if not os.path.isdir(path):
                raise FileNotFoundError(f'找不到模型資料夾：{path}')

        on_append(f'載入 ws 模型：{ws_path}')
        ws = CkipWordSegmenter(model_name=ws_path, device=DEVICE)

        on_append(f'載入 pos 模型：{pos_path}')
        pos = CkipPosTagger(model_name=pos_path, device=DEVICE)

        on_done((ws, pos))

    except Exception as e:
        on_error(str(e))


def start_load_model(model_dir: str, on_append, on_done, on_error):
    """啟動模型載入背景執行緒。"""
    t = threading.Thread(
        target=load_model_worker,
        args=(model_dir, on_append, on_done, on_error),
        daemon=True,
    )
    t.start()
    return t


def run_tagger(ckip_ws, ckip_pos, input_files, output_folder, on_append, on_done, on_error):
    """
    背景執行緒：對每個輸入 xlsx 執行斷詞並存檔。

    Callbacks
    ---------
    on_append(msg)   — log 輸出
    on_done()        — 全部完成
    on_error(msg)    — 發生錯誤
    """
    def _worker():
        try:
            total = len(input_files)
            for i, file_path in enumerate(input_files):
                on_append(f'[{i+1}/{total}] 處理：{os.path.basename(file_path)}')
                try:
                    df = pd.read_excel(file_path)
                    df = df.dropna(subset=['content'])
                    df['sent_seq'] = range(1, len(df) + 1)

                    sentences   = [str(r['content']) for _, r in df.iterrows()]
                    ws_results  = ckip_ws(sentences)
                    pos_results = ckip_pos(ws_results)

                    tagger_df_list = []
                    for sent_seq, words, tags in zip(df['sent_seq'], ws_results, pos_results):
                        tagger_df = pd.DataFrame({
                            'sent_seq': [sent_seq] * len(words),
                            'word':     words,
                            'pos':      tags,
                        })
                        tagger_df['sentsword_seq'] = range(1, len(tagger_df) + 1)
                        tagger_df_list.append(tagger_df)

                    if tagger_df_list:
                        result_df = pd.concat(tagger_df_list, ignore_index=True)
                        out_path  = os.path.join(output_folder, os.path.basename(file_path))
                        result_df.to_excel(out_path, index=False)
                        on_append(f'  -> 已存：{out_path}')

                except Exception as e:
                    on_append(f'  錯誤：{e}')

            on_append('斷詞已完成\n')
            on_done()

        except Exception as e:
            on_error(str(e))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
