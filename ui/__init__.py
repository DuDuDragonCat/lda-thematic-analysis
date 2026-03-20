import sys
import os


def get_app_dir() -> str:
    """回傳軟體執行時的工作根目錄（用於存放設定檔、預設瀏覽位置等）。

    打包後（PyInstaller frozen）：
      - macOS .app：sys.executable = .app/Contents/MacOS/exe → 往上 3 層 = .app bundle 本身
      - Windows .exe：sys.executable = 直接是 .exe 本身 → 往上 1 層 = exe 所在資料夾

    開發模式（直接執行 main.py）：
      - ui/__init__.py 在 <project>/ui/，往上一層即專案根目錄
    """
    if getattr(sys, 'frozen', False):
        if sys.platform == 'darwin':
            # sys.executable = …/LDA分析工具.app/Contents/MacOS/LDA分析工具
            # dirname × 1 → …/LDA分析工具.app/Contents/MacOS
            # dirname × 2 → …/LDA分析工具.app/Contents
            # dirname × 3 → …/LDA分析工具.app  （.app bundle 本身，狀態檔存於此）
            return os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))
        else:
            # Windows / Linux：sys.executable 就是 .exe
            # dirname × 1 → exe 所在資料夾（狀態檔存於此，與 exe 並列）
            return os.path.dirname(sys.executable)
    else:
        # ui/__init__.py 在 <project>/ui/，往上一層即專案根目錄
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
