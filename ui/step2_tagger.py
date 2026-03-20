# ui/step2_tagger.py
# 步驟2：載入 CKIP 模型並執行斷詞的 UI 分頁

import glob
import os
import queue
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

from workers.tagger_worker import WS_FOLDER, POS_FOLDER, start_load_model, run_tagger
from ui import get_app_dir


class Step2TaggerFrame(ctk.CTkFrame):
    """步驟2 — 載入模型並斷詞分頁。"""

    def __init__(self, master, state, **kwargs):
        super().__init__(master, **kwargs)
        self._state = state
        self._queue: queue.Queue = queue.Queue()
        self._busy = False
        self._ckip_ws = None
        self._ckip_pos = None
        self._input_files = []
        self._output_folder = None
        self._build_ui()
        self._on_model_folder_change()  # 用載入的路徑初始化按鈕狀態
        self._on_io_change()
        self._poll()

    # ── UI 建構 ────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)

        desc = ctk.CTkLabel(
            self,
            text=(
                "步驟 2：載入斷詞模型並對 Excel 檔進行中文斷詞\n"
                f"模型資料夾須包含：{WS_FOLDER}、{POS_FOLDER} 兩個子資料夾\n"
                "輸入的 Excel 檔需有 'content' 欄位"
            ),
            justify='left', anchor='w',
        )
        desc.grid(row=0, column=0, padx=16, pady=(16, 8), sticky='w')

        # 模型資料夾（Step1 ↔ Step2 共享）
        self._model_folder = self._make_folder_row(row=1, label='模型資料夾：',
                                                   var=self._state.model_folder)

        # 載入模型按鈕
        self._btn_load = ctk.CTkButton(
            self, text='載入模型', width=120, state='disabled',
            command=self._load_model,
        )
        self._btn_load.grid(row=2, column=0, padx=16, pady=4, sticky='w')

        # 模型狀態
        self._model_status = ctk.CTkLabel(
            self, text='模型狀態：未載入', text_color='gray', anchor='w')
        self._model_status.grid(row=3, column=0, padx=16, pady=(0, 4), sticky='w')

        ctk.CTkFrame(self, height=1, fg_color='gray40').grid(
            row=4, column=0, padx=16, pady=6, sticky='ew')

        # 輸入 / 輸出資料夾
        self._input_folder_var  = self._make_folder_row(row=5, label='輸入資料夾（.xlsx）：',
                                                        var=self._state.step2_input)
        self._output_folder_var = self._make_folder_row(row=6, label='輸出資料夾：',
                                                        var=self._state.tagged_folder)  # Step2↔3/4 共享

        # 執行按鈕
        self._btn_run = ctk.CTkButton(
            self, text='執行斷詞', width=120, state='disabled',
            command=self._run_tagger,
        )
        self._btn_run.grid(row=7, column=0, padx=16, pady=8, sticky='w')

        # Log
        ctk.CTkLabel(self, text='執行 Log：', anchor='w').grid(
            row=8, column=0, padx=16, pady=(4, 2), sticky='w')
        self._log_box = ctk.CTkTextbox(
            self, font=('Courier New', 12), state='disabled', wrap='none')
        self._log_box.grid(row=9, column=0, padx=16, pady=(0, 16), sticky='nsew')
        self.grid_rowconfigure(9, weight=1)

        # 資料夾變更追蹤
        self._model_folder.trace_add('write', self._on_model_folder_change)
        self._input_folder_var.trace_add('write', self._on_io_change)
        self._output_folder_var.trace_add('write', self._on_io_change)

    def _make_folder_row(self, row: int, label: str,
                         var: tk.StringVar | None = None) -> tk.StringVar:
        frame = ctk.CTkFrame(self, fg_color='transparent')
        frame.grid(row=row, column=0, padx=16, pady=3, sticky='ew')
        frame.grid_columnconfigure(1, weight=1)
        if var is None:
            var = tk.StringVar()
        ctk.CTkLabel(frame, text=label, width=180, anchor='w').grid(row=0, column=0)
        ctk.CTkEntry(frame, textvariable=var, width=340).grid(
            row=0, column=1, sticky='ew', padx=(0, 8))
        ctk.CTkButton(
            frame, text='瀏覽', width=70,
            command=lambda v=var: self._browse(v),
        ).grid(row=0, column=2)
        return var

    # ── 事件 ────────────────────────────────────────────────────────────────

    def _browse(self, var: tk.StringVar):
        folder = filedialog.askdirectory(initialdir=get_app_dir())
        if folder:
            var.set(folder)

    def _on_model_folder_change(self, *_):
        has = bool(self._model_folder.get().strip())
        self._btn_load.configure(state='normal' if has and not self._busy else 'disabled')

    def _on_io_change(self, *_):
        in_ok = bool(self._input_folder_var.get().strip())
        out_ok = bool(self._output_folder_var.get().strip())
        model_ok = self._ckip_ws is not None
        ok = in_ok and out_ok and model_ok and not self._busy
        self._btn_run.configure(state='normal' if ok else 'disabled')

    def _load_model(self):
        folder = self._model_folder.get().strip()
        if not folder:
            return
        self._busy = True
        self._btn_load.configure(state='disabled')
        self._btn_run.configure(state='disabled')
        self._model_status.configure(text='模型狀態：載入中…', text_color='orange')
        self._append_log(f'載入模型資料夾：{folder}')

        start_load_model(
            model_dir=folder,
            on_append=lambda msg: self._queue.put(('append', msg)),
            on_done=lambda pair: self._queue.put(('model_done', pair)),
            on_error=lambda err: self._queue.put(('model_error', err)),
        )

    def _run_tagger(self):
        in_folder = self._input_folder_var.get().strip()
        out_folder = self._output_folder_var.get().strip()

        if not in_folder or not out_folder:
            messagebox.showwarning('提示', '請選擇輸入與輸出資料夾。')
            return
        if in_folder == out_folder:
            messagebox.showwarning('提示', '請選擇不同的輸出資料夾。')
            return

        files = [f for f in glob.glob(in_folder + '/*.xlsx')
                 if not os.path.basename(f).startswith('~$')]
        if not files:
            messagebox.showwarning('提示', '輸入資料夾內沒有 .xlsx 檔案。')
            return

        self._busy = True
        self._btn_run.configure(state='disabled')
        self._btn_load.configure(state='disabled')
        self._append_log(f'讀取資料夾：{in_folder}（共 {len(files)} 筆）')

        run_tagger(
            ckip_ws=self._ckip_ws,
            ckip_pos=self._ckip_pos,
            input_files=files,
            output_folder=out_folder,
            on_append=lambda msg: self._queue.put(('append', msg)),
            on_done=lambda: self._queue.put(('tagger_done', None)),
            on_error=lambda err: self._queue.put(('error', err)),
        )

    # ── log 工具 ────────────────────────────────────────────────────────────

    def _append_log(self, msg: str):
        self._log_box.configure(state='normal')
        self._log_box.insert('end', msg + '\n')
        self._log_box.see('end')
        self._log_box.configure(state='disabled')

    # ── 輪詢 Queue ─────────────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == 'append':
                    self._append_log(payload)
                elif kind == 'model_done':
                    self._ckip_ws, self._ckip_pos = payload
                    self._busy = False
                    self._model_status.configure(
                        text='模型狀態：已載入', text_color='green')
                    test = self._ckip_ws(['模型載入完成。'])
                    self._append_log(f'測試斷詞：{" ".join(test[0])}\n')
                    self._on_model_folder_change()
                    self._on_io_change()
                elif kind == 'model_error':
                    self._busy = False
                    self._ckip_ws = self._ckip_pos = None
                    self._model_status.configure(
                        text='模型狀態：載入失敗', text_color='red')
                    self._append_log(f'模型載入失敗：{payload}\n')
                    self._on_model_folder_change()
                elif kind == 'tagger_done':
                    self._busy = False
                    self._on_io_change()
                    self._on_model_folder_change()
                elif kind == 'error':
                    self._busy = False
                    self._append_log(f'錯誤：{payload}\n')
                    self._on_io_change()
                    self._on_model_folder_change()
        except queue.Empty:
            pass
        self.after(100, self._poll)
