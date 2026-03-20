# workers/download_worker.py
# 下載 CKIP bert-base 模型的後台邏輯（從 script/download.py 提取）

import io
import os
import re
import sys
import threading

# 模組層級 import：在 splash 啟動期間（主執行緒）載入，
# 避免在背景執行緒 lazy import 時觸發 multiprocessing spawn，
# 進而重新執行 main.py 並開啟多餘的 splash 視窗。
from huggingface_hub import snapshot_download

MODELS = [
    ('ckiplab/bert-base-chinese-ws',  'bert-base-chinese-ws'),
    ('ckiplab/bert-base-chinese-pos', 'bert-base-chinese-pos'),
]


class _GuiWriter(io.TextIOBase):
    """攔截 tqdm 的 stderr/stdout，逐字元解析 \\r/\\n 後呼叫回呼函式。"""

    # 過濾 ANSI escape codes（如 \x1b[A cursor-up、\x1b[K 清行等）
    _ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

    def __init__(self, on_overwrite, on_append):
        self._on_overwrite = on_overwrite
        self._on_append = on_append
        self._buf = ''

    def write(self, s):
        s = self._ANSI_RE.sub('', s)   # 移除 ANSI 控制碼
        for ch in s:
            if ch == '\r':
                if self._buf:              # 空 buf 不發 overwrite，避免覆蓋有效行為空白
                    self._on_overwrite(self._buf)
                self._buf = ''
            elif ch == '\n':
                if self._buf.strip():  # 略過 tqdm 預留空間產生的空白行
                    self._on_append(self._buf)
                self._buf = ''
            else:
                self._buf += ch
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return True   # 讓 tqdm 維持進度條格式


def download_all_worker(base_dir: str, on_append, on_overwrite, on_done, on_error):
    """
    背景執行緒：依序下載所有模型。

    Callbacks
    ---------
    on_append(msg: str)      — 在 log 追加一行
    on_overwrite(msg: str)   — 覆寫 log 最後一行（tqdm \\r 行為）
    on_done(base_dir: str)   — 全部下載完成
    on_error(msg: str)       — 發生錯誤
    """
    writer = _GuiWriter(on_overwrite=on_overwrite, on_append=on_append)
    old_stdout, old_stderr = sys.stdout, sys.stderr

    try:
        sys.stdout = sys.stderr = writer

        for repo_id, folder_name in MODELS:
            save_dir = os.path.join(base_dir, folder_name)
            on_append(f'=== 下載：{repo_id} ===')
            snapshot_download(repo_id=repo_id, local_dir=save_dir)
            on_append(f'完成：{save_dir}')

        on_done(base_dir)

    except Exception as e:
        on_error(str(e))

    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def start_download(base_dir: str, on_append, on_overwrite, on_done, on_error):
    """啟動下載背景執行緒。"""
    t = threading.Thread(
        target=download_all_worker,
        args=(base_dir, on_append, on_overwrite, on_done, on_error),
        daemon=True,
    )
    t.start()
    return t
