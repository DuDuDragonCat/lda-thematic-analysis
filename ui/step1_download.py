# ui/step1_download.py
# 步驟1：下載 CKIP 斷詞模型的 UI 分頁

import queue
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk

from workers.download_worker import MODELS, start_download
from ui import get_app_dir


class Step1DownloadFrame(ctk.CTkFrame):
    """步驟1 — 下載斷詞模型分頁。"""

    def __init__(self, master, state, **kwargs):
        super().__init__(master, **kwargs)
        self._state = state
        self._queue: queue.Queue = queue.Queue()
        self._busy = False
        self._build_ui()
        self._on_folder_change()  # 用載入的路徑初始化按鈕狀態
        self._poll()

    # ── UI 建構 ────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)

        # 說明
        desc = ctk.CTkLabel(
            self,
            text=(
                "步驟 1：下載 CKIP 中文斷詞模型\n"
                "選擇一個資料夾作為模型存放位置，再按「開始下載」。\n"
                f"將會下載：{', '.join(m[0] for m in MODELS)}"
            ),
            justify='left',
            anchor='w',
        )
        desc.grid(row=0, column=0, padx=16, pady=(16, 8), sticky='w')

        # 資料夾選擇
        folder_frame = ctk.CTkFrame(self, fg_color='transparent')
        folder_frame.grid(row=1, column=0, padx=16, pady=4, sticky='ew')
        folder_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(folder_frame, text='存放資料夾：').grid(row=0, column=0, padx=(0, 8))

        self._folder_var = self._state.model_folder   # Step1 ↔ Step2 共享
        self._folder_entry = ctk.CTkEntry(
            folder_frame, textvariable=self._folder_var, width=400)
        self._folder_entry.grid(row=0, column=1, sticky='ew', padx=(0, 8))

        ctk.CTkButton(
            folder_frame, text='瀏覽', width=80,
            command=self._browse_folder,
        ).grid(row=0, column=2)

        # 下載按鈕
        self._btn_download = ctk.CTkButton(
            self, text='開始下載', width=140, state='disabled',
            command=self._start_download,
        )
        self._btn_download.grid(row=2, column=0, padx=16, pady=8, sticky='w')

        # Log 區域
        ctk.CTkLabel(self, text='下載進度：', anchor='w').grid(
            row=3, column=0, padx=16, pady=(8, 2), sticky='w')

        self._log_box = ctk.CTkTextbox(
            self, font=('Courier New', 12), state='disabled', wrap='none')
        self._log_box.grid(row=4, column=0, padx=16, pady=(0, 16), sticky='nsew')
        self.grid_rowconfigure(4, weight=1)

        self._folder_var.trace_add('write', self._on_folder_change)

    # ── 事件處理 ────────────────────────────────────────────────────────────

    def _browse_folder(self):
        folder = filedialog.askdirectory(title='選擇模型存放資料夾', initialdir=get_app_dir())
        if folder:
            self._folder_var.set(folder)

    def _on_folder_change(self, *_):
        has_folder = bool(self._folder_var.get().strip())
        self._btn_download.configure(state='normal' if has_folder and not self._busy else 'disabled')

    def _start_download(self):
        folder = self._folder_var.get().strip()
        if not folder:
            self._append_log('請先選擇存放資料夾。')
            return
        self._busy = True
        self._btn_download.configure(state='disabled')
        self._append_log(f'存放位置：{folder}')
        for repo_id, fname in MODELS:
            self._append_log(f'  將下載：{repo_id} → {fname}')
        self._append_log('')

        start_download(
            base_dir=folder,
            on_append=lambda msg: self._queue.put(('append', msg)),
            on_overwrite=lambda msg: self._queue.put(('overwrite', msg)),
            on_done=lambda bd: self._queue.put(('done', bd)),
            on_error=lambda err: self._queue.put(('error', err)),
        )

    # ── log 工具 ────────────────────────────────────────────────────────────

    def _append_log(self, msg: str):
        self._log_box.configure(state='normal')
        self._log_box.insert('end', msg + '\n')
        self._log_box.see('end')
        self._log_box.configure(state='disabled')

    def _overwrite_last_line(self, msg: str):
        self._log_box.configure(state='normal')
        self._log_box.delete('end-2l', 'end-1l')
        self._log_box.insert('end-1c', msg + '\n')
        self._log_box.see('end')
        self._log_box.configure(state='disabled')

    # ── 輪詢 Queue ─────────────────────────────────────────────────────────

    def _poll(self):
        # tqdm 每個 chunk 就發一次 overwrite，100 ms 內可能累積數十條。
        # 先排空 queue，overwrite 只保留最後一條，避免爆發性 tkinter 操作造成卡頓。
        pending_overwrite = None
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == 'append':
                    if pending_overwrite is not None:       # overwrite 先刷出
                        self._overwrite_last_line(pending_overwrite)
                        pending_overwrite = None
                    self._append_log(payload)
                elif kind == 'overwrite':
                    pending_overwrite = payload             # 只保留最新一條
                elif kind == 'done':
                    pending_overwrite = None
                    self._busy = False
                    self._append_log('=== 全部下載完成 ===')
                    for _, fname in MODELS:
                        import os
                        self._append_log(f'  {os.path.join(payload, fname)}')
                    self._on_folder_change()
                elif kind == 'error':
                    pending_overwrite = None
                    self._busy = False
                    self._append_log(f'下載失敗：{payload}')
                    self._on_folder_change()
        except queue.Empty:
            pass
        if pending_overwrite is not None and pending_overwrite.strip():  # 非空才刷出
            self._overwrite_last_line(pending_overwrite)
        self.after(50, self._poll)                          # 50 ms 更新頻率更順暢
