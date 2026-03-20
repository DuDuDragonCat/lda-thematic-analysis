# ui/step4_fitlda.py
# 步驟4：LDA 適配與主題分配的 UI 分頁

import glob
import os
import queue
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

from workers.fitlda_worker import DEFAULT, safe_int, run_fitlda
from ui import get_app_dir


class Step4FitLDAFrame(ctk.CTkFrame):
    """步驟4 — LDA 適配與主題分配分頁。"""

    def __init__(self, master, state, **kwargs):
        super().__init__(master, **kwargs)
        self._state = state
        self._queue: queue.Queue = queue.Queue()
        self._busy = False
        self._build_ui()
        self._on_io_change()  # 用載入的路徑初始化按鈕狀態
        self._poll()

    # ── UI 建構 ────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)

        desc = ctk.CTkLabel(
            self,
            text=(
                "步驟 4：以指定主題數（K）適配 LDA，並將段落與文章指派到最佳主題\n"
                "輸入為步驟2斷詞後的資料夾；K 值請參考步驟3的指標圖選擇\n"
                "輸出：TopicPart（段落主題分布）、WordTopic（詞-主題分布）、TopicDoc（文章主題分配）"
            ),
            justify='left', anchor='w',
        )
        desc.grid(row=0, column=0, padx=16, pady=(16, 8), sticky='w')

        # 資料夾選擇（Step2 輸出 ↔ Step3/4 輸入；Step3 ↔ Step4 輸出 共享）
        self._input_var  = self._make_folder_row(row=1, label='輸入資料夾（斷詞後）：',
                                                 var=self._state.tagged_folder)
        self._output_var = self._make_folder_row(row=2, label='輸出資料夾：',
                                                 var=self._state.lda_output)

        # 參數區域
        param_outer = ctk.CTkFrame(self, fg_color='transparent')
        param_outer.grid(row=3, column=0, padx=16, pady=8, sticky='ew')

        param_left  = ctk.CTkFrame(param_outer, fg_color='transparent')
        param_right = ctk.CTkFrame(param_outer, fg_color='transparent')
        param_left.pack(side='left', padx=(0, 32))
        param_right.pack(side='left')

        # Step3 ↔ Step4 共享 LDA 參數
        self._seed_var     = self._make_param(param_left,  row=0, label='SEED：',             var=self._state.seed)
        self._burnin_var   = self._make_param(param_left,  row=1, label='Burn In：',           var=self._state.burnin)
        self._iter_var     = self._make_param(param_left,  row=2, label='Iteration：',         var=self._state.iteration)
        self._thin_var     = self._make_param(param_left,  row=3, label='Thin Interval：',     var=self._state.thin)
        # Step4 私有參數（持久化）
        self._topicnum_var = self._make_param(param_right, row=0, label='LDA Topic Number（K）：', var=self._state.step4_topicnum)
        self._wordsnum_var = self._make_param(param_right, row=1, label='每段落詞數：',             var=self._state.step4_wordsnum)

        # 按鈕列
        btn_frame = ctk.CTkFrame(self, fg_color='transparent')
        btn_frame.grid(row=4, column=0, padx=16, pady=4, sticky='w')

        ctk.CTkButton(btn_frame, text='重置參數', width=100,
                      command=self._reset_params).pack(side='left', padx=(0, 8))
        self._btn_run = ctk.CTkButton(
            btn_frame, text='LDA 模型適配', width=140,
            command=self._run, state='disabled')
        self._btn_run.pack(side='left')

        # Log
        ctk.CTkLabel(self, text='執行 Log：', anchor='w').grid(
            row=5, column=0, padx=16, pady=(8, 2), sticky='w')
        self._log_box = ctk.CTkTextbox(
            self, font=('Courier New', 12), state='disabled', wrap='none')
        self._log_box.grid(row=6, column=0, padx=16, pady=(0, 16), sticky='nsew')
        self.grid_rowconfigure(6, weight=1)

        self._input_var.trace_add('write', self._on_io_change)
        self._output_var.trace_add('write', self._on_io_change)

    def _make_folder_row(self, row: int, label: str,
                         var: tk.StringVar | None = None) -> tk.StringVar:
        frame = ctk.CTkFrame(self, fg_color='transparent')
        frame.grid(row=row, column=0, padx=16, pady=3, sticky='ew')
        frame.grid_columnconfigure(1, weight=1)
        if var is None:
            var = tk.StringVar()
        ctk.CTkLabel(frame, text=label, width=200, anchor='w').grid(row=0, column=0)
        ctk.CTkEntry(frame, textvariable=var, width=320).grid(
            row=0, column=1, sticky='ew', padx=(0, 8))
        ctk.CTkButton(frame, text='瀏覽', width=70,
                      command=lambda v=var: self._browse(v)).grid(row=0, column=2)
        return var

    def _make_param(self, parent, row: int, label: str,
                    default: str = '', var: tk.StringVar | None = None) -> tk.StringVar:
        if var is None:
            var = tk.StringVar(value=default)
        ctk.CTkLabel(parent, text=label, width=180, anchor='w').grid(
            row=row, column=0, pady=2, sticky='w')
        ctk.CTkEntry(parent, textvariable=var, width=80).grid(
            row=row, column=1, pady=2, padx=(0, 8))
        return var

    # ── 事件 ────────────────────────────────────────────────────────────────

    def _browse(self, var: tk.StringVar):
        folder = filedialog.askdirectory(initialdir=get_app_dir())
        if folder:
            var.set(folder)

    def _on_io_change(self, *_):
        in_ok  = bool(self._input_var.get().strip())
        out_ok = bool(self._output_var.get().strip())
        self._btn_run.configure(
            state='normal' if in_ok and out_ok and not self._busy else 'disabled')

    def _reset_params(self):
        self._state.reset_lda_params()   # 重置 Step3/4 共享 LDA 參數
        self._topicnum_var.set(DEFAULT['topicnum'])
        self._wordsnum_var.set(DEFAULT['words_num'])
        self._append_log('已重置參數。')

    def _run(self):
        in_folder  = self._input_var.get().strip()
        out_folder = self._output_var.get().strip()

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

        seed      = safe_int(self._seed_var.get(),     777)
        burnin    = safe_int(self._burnin_var.get(),   500)
        itr       = safe_int(self._iter_var.get(),     1000)
        thin      = safe_int(self._thin_var.get(),     100)
        topicnum  = safe_int(self._topicnum_var.get(), 2)
        words_num = safe_int(self._wordsnum_var.get(), 50)

        self._busy = True
        self._btn_run.configure(state='disabled')
        self._append_log('開始 LDA 模型適配')

        run_fitlda(
            input_files=files,
            output_folder=out_folder,
            k=topicnum,
            seed=seed, burn_in=burnin, iteration=itr, thin=thin,
            words_num=words_num,
            on_append=lambda msg: self._queue.put(('append', msg)),
            on_done=lambda: self._queue.put(('done', None)),
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
                elif kind == 'done':
                    self._busy = False
                    self._on_io_change()
                elif kind == 'error':
                    self._busy = False
                    self._append_log(f'LDA 計算失敗：{payload}\n')
                    self._on_io_change()
        except queue.Empty:
            pass
        self.after(100, self._poll)
