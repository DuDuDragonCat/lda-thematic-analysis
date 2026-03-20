# ui/app_state.py
# 跨步驟共享狀態管理，並自動持久化使用者設定。

import json
import os
import tkinter as tk

from ui import get_app_dir

# 從各 worker 匯入預設值，以 worker 為唯一真實來源（single source of truth）。
# 共享 LDA 參數（seed/burnin/iteration/thin）兩個 worker 定義相同，以 findbest 為準。
from workers.findbest_worker import DEFAULT as _FB_DEFAULT
from workers.fitlda_worker   import DEFAULT as _FL_DEFAULT

_STATE_FILENAME = 'lda_app_state.json'

_DEFAULTS = {
    # ── 資料夾路徑（無對應 worker 常數）────────────────────────────────
    'model_folder':   '',
    'tagged_folder':  '',
    'lda_output':     '',
    'step2_input':    '',
    # ── LDA 共享參數：Step3 & Step4（以 findbest_worker.DEFAULT 為準）──
    'seed':           _FB_DEFAULT['seed'],
    'burnin':         _FB_DEFAULT['burnin'],
    'iteration':      _FB_DEFAULT['iteration'],
    'thin':           _FB_DEFAULT['thin'],
    # ── Step3 私有參數（findbest_worker.DEFAULT）────────────────────────
    'step3_min':      _FB_DEFAULT['min_topic'],
    'step3_max':      _FB_DEFAULT['max_topic'],
    'step3_per':      _FB_DEFAULT['per_topic'],
    # ── Step4 私有參數（fitlda_worker.DEFAULT）──────────────────────────
    'step4_topicnum': _FL_DEFAULT['topicnum'],
    'step4_wordsnum': _FL_DEFAULT['words_num'],
}


class AppState:
    """
    持有所有跨步驟共享的 tk.StringVar，並在每次變更時自動儲存到
    <app_dir>/lda_app_state.json，啟動時自動載入。

    共享關係
    --------
    model_folder  : Step1 ↔ Step2（模型資料夾）
    tagged_folder : Step2 輸出 ↔ Step3/4 輸入（斷詞後資料夾）
    lda_output    : Step3 ↔ Step4（LDA 輸出資料夾）
    seed / burnin / iteration / thin : Step3 ↔ Step4（LDA 參數）
    """

    def __init__(self):
        # ── 共享 vars ─────────────────────────────────────────────────────────
        self.model_folder   = tk.StringVar()   # Step1 & Step2 模型資料夾
        self.tagged_folder  = tk.StringVar()   # Step2 輸出 = Step3/4 輸入
        self.lda_output     = tk.StringVar()   # Step3 & Step4 LDA 輸出

        self.seed           = tk.StringVar()   # Step3 & Step4 共享
        self.burnin         = tk.StringVar()
        self.iteration      = tk.StringVar()
        self.thin           = tk.StringVar()

        # ── 各步驟私有（僅持久化）─────────────────────────────────────────────
        self.step2_input    = tk.StringVar()   # Step2 輸入資料夾（原始 xlsx）
        self.step3_min      = tk.StringVar()
        self.step3_max      = tk.StringVar()
        self.step3_per      = tk.StringVar()
        self.step4_topicnum = tk.StringVar()
        self.step4_wordsnum = tk.StringVar()

        # 載入上次狀態（先載入再掛 trace，避免載入時觸發大量 save）
        self._load()

        # 任何 var 變更時自動儲存
        for var in self._all_vars():
            var.trace_add('write', self._save)

    # ── 公開工具 ────────────────────────────────────────────────────────────

    def reset_lda_params(self):
        """將 LDA 共享參數重置為預設值。"""
        self.seed.set(_DEFAULTS['seed'])
        self.burnin.set(_DEFAULTS['burnin'])
        self.iteration.set(_DEFAULTS['iteration'])
        self.thin.set(_DEFAULTS['thin'])

    # ── 內部 ────────────────────────────────────────────────────────────────

    def _all_vars(self):
        return [v for v in vars(self).values() if isinstance(v, tk.StringVar)]

    def _state_path(self) -> str:
        return os.path.join(get_app_dir(), _STATE_FILENAME)

    def _load(self):
        try:
            with open(self._state_path(), 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            data = {}

        for key, default in _DEFAULTS.items():
            getattr(self, key).set(data.get(key, default))

    def _save(self, *_):
        try:
            data = {
                k: v.get()
                for k, v in vars(self).items()
                if isinstance(v, tk.StringVar)
            }
            with open(self._state_path(), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass
