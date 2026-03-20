# main.py — LDA 整合 GUI 應用程式入口點
# 使用 CustomTkinter 建立分頁式介面，整合下載、斷詞、K 選擇、LDA 適配四個步驟。

import sys
import os

# PyInstaller 打包後確保 sys.path 正確
if getattr(sys, 'frozen', False):
    _base = sys._MEIPASS
    sys.path.insert(0, _base)

    # console=False ビルドでは sys.stdout / sys.stderr が None になる。
    # huggingface_hub / tqdm / tokenizers が内部スレッドから書き込むと
    # AttributeError → tkinter エラーハンドラが余分なウィンドウを開く原因になるため、
    # None の代わりに devnull を指しておく。
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')

import customtkinter as ctk


class LDAApp(ctk.CTk):
    """整合 LDA 分析流程的主視窗，啟動時顯示載入進度。"""

    _LOAD_STEPS = 4   # 與 _load_modules() 中 _splash_step() 呼叫次數一致

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode('System')
        ctk.set_default_color_theme('blue')

        self.title('LDA 中文主題模型分析工具')
        self.geometry('420x130')
        self.resizable(False, False)

        # ① 先顯示載入畫面（使用同一個 Tk 根，不另起 tk.Tk()）
        self._build_splash()
        self.update()

        # ② 依序載入重型模組，更新進度
        self._load_modules()

        # ③ 換成完整主 UI
        self.geometry('900x720')
        self.minsize(800, 600)
        self.resizable(True, True)
        self._build_ui()

    # ── 載入畫面 ──────────────────────────────────────────────────────────────

    def _build_splash(self):
        self._splash_frame = ctk.CTkFrame(self)
        self._splash_frame.pack(fill='both', expand=True)

        ctk.CTkLabel(
            self._splash_frame,
            text='LDA 中文主題模型分析工具',
            font=ctk.CTkFont(size=14, weight='bold'),
        ).pack(pady=(22, 4))

        self._splash_status = ctk.CTkLabel(
            self._splash_frame,
            text='初始化中…',
            font=ctk.CTkFont(size=10),
            text_color='gray',
        )
        self._splash_status.pack(pady=2)

        self._splash_bar = ctk.CTkProgressBar(self._splash_frame, width=370)
        self._splash_bar.pack(pady=10, padx=24)
        self._splash_bar.set(0)

    def _splash_step(self, step: int, text: str):
        """更新狀態文字並推進進度條。"""
        self._splash_status.configure(text=text)
        self._splash_bar.set(step / self._LOAD_STEPS)
        self.update()

    # ── 依序載入重型模組 ──────────────────────────────────────────────────────

    def _load_modules(self):
        self._splash_step(1, '載入下載 / 斷詞模組…')
        from ui.step1_download import Step1DownloadFrame
        from ui.step2_tagger   import Step2TaggerFrame

        self._splash_step(2, '載入 tomotopy / scipy…')
        from ui.step3_findbest import Step3FindBestFrame

        self._splash_step(3, '載入 LDA 分析模組…')
        from ui.step4_fitlda   import Step4FitLDAFrame

        self._splash_step(4, '啟動應用程式…')

        # 暫存供 _build_ui() 使用
        self._frame_classes = (
            Step1DownloadFrame,
            Step2TaggerFrame,
            Step3FindBestFrame,
            Step4FitLDAFrame,
        )

    # ── 主 UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # 移除載入畫面
        self._splash_frame.destroy()
        del self._splash_frame, self._splash_status, self._splash_bar

        # 建立共享狀態（含持久化）
        from ui.app_state import AppState
        self._state = AppState()

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # 標題列
        ctk.CTkLabel(
            self,
            text='LDA 中文主題模型分析工具',
            font=ctk.CTkFont(size=18, weight='bold'),
        ).grid(row=0, column=0, padx=20, pady=(16, 8), sticky='w')

        # 分頁
        self._tabview = ctk.CTkTabview(self)
        self._tabview.grid(row=1, column=0, padx=12, pady=(0, 12), sticky='nsew')

        tab_names = [
            '步驟 1  下載斷詞模型',
            '步驟 2  執行斷詞',
            '步驟 3  選擇最佳 K',
            '步驟 4  LDA 適配',
        ]
        for name in tab_names:
            self._tabview.add(name)

        Step1, Step2, Step3, Step4 = self._frame_classes
        del self._frame_classes

        def _embed(tab_name, FrameClass):
            tab = self._tabview.tab(tab_name)
            tab.grid_rowconfigure(0, weight=1)
            tab.grid_columnconfigure(0, weight=1)
            frame = FrameClass(tab, state=self._state)
            frame.grid(row=0, column=0, sticky='nsew')
            return frame

        self._step1 = _embed(tab_names[0], Step1)
        self._step2 = _embed(tab_names[1], Step2)
        self._step3 = _embed(tab_names[2], Step3)
        self._step4 = _embed(tab_names[3], Step4)


def main():
    app = LDAApp()
    app.mainloop()


if __name__ == '__main__':
    # multiprocessing spawn（macOS 預設）衍生子程序時會重新執行 main.py，
    # freeze_support() 能讓子程序在執行 main() 之前提早退出。
    import multiprocessing
    multiprocessing.freeze_support()
    main()
