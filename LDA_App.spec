# LDA_App.spec — PyInstaller 打包設定（支援 macOS / Windows，onefile 模式）
# 使用方式：
#   uv run python -m PyInstaller LDA_App.spec
#
# macOS 產出：dist/LDA分析工具.app  （單一 .app bundle，無額外資料夾）
# Windows 產出：dist/LDA分析工具.exe （單一執行檔）

import sys
import os
import glob as _glob
from PyInstaller.utils.hooks import collect_data_files

IS_MAC = sys.platform == 'darwin'
IS_WIN = sys.platform == 'win32'

# ── customtkinter 資源路徑 ────────────────────────────────────────────────────
try:
    import customtkinter
    CTK_PATH = os.path.dirname(customtkinter.__file__)
except ImportError:
    CTK_PATH = None  # 打包前請先安裝 customtkinter

datas = []
if CTK_PATH:
    datas += [(CTK_PATH, 'customtkinter')]

datas += collect_data_files('ckip_transformers')
datas += collect_data_files('matplotlib')   # 字型、style 等執行期資源

# tomotopy SIMD 原生擴充模組（在 site-packages 根目錄，需明確加入）
_tp_binaries = []
if IS_WIN:
    import site
    for _sp in site.getsitepackages():
        _found = _glob.glob(os.path.join(_sp, '_tomotopy*.pyd'))
        _tp_binaries += [(_f, '.') for _f in _found]

# ── Analysis ──────────────────────────────────────────────────────────────────
_analysis_kwargs = dict(
    scripts=['main.py'],
    pathex=['.'],
    binaries=_tp_binaries,
    datas=datas,
    hiddenimports=[
        # customtkinter
        'customtkinter',
        # ckip-transformers
        'ckip_transformers',
        'ckip_transformers.nlp',
        'ckip_transformers.nlp.core',
        'ckip_transformers.model',
        # transformers / torch
        'transformers',
        'torch',
        'tokenizers',
        # tomotopy（包含 SIMD 動態載入的原生模組）
        'tomotopy',
        '_tomotopy',
        '_tomotopy_avx512',
        '_tomotopy_avx2',
        '_tomotopy_sse2',
        '_tomotopy_none',
        # scipy
        'scipy.linalg',
        'scipy.special',
        # huggingface_hub
        'huggingface_hub',
        # matplotlib
        'matplotlib',
        'matplotlib.backends.backend_agg',
        # standard library
        'queue',
        'threading',
        'glob',
        'openpyxl',
        # UI / worker 模組
        'ui',
        'ui.step1_download',
        'ui.step2_tagger',
        'ui.step3_findbest',
        'ui.step4_fitlda',
        'workers.download_worker',
        'workers.tagger_worker',
        'workers.findbest_worker',
        'workers.fitlda_worker',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Windows 專用參數（macOS 不支援，不傳入）
if IS_WIN:
    _analysis_kwargs['win_no_prefer_redirects'] = False
    _analysis_kwargs['win_private_assemblies']  = False

a = Analysis(**_analysis_kwargs)

pyz = PYZ(a.pure, a.zipped_data)

# ── 平台圖示 ──────────────────────────────────────────────────────────────────
# macOS: .icns　Windows: .ico　未設定時傳 None
_icon = None
# if IS_MAC:  _icon = 'assets/icon.icns'
# if IS_WIN:  _icon = 'assets/icon.ico'

# ── EXE（onefile 模式：binaries / datas 直接內嵌，不使用 COLLECT）────────────
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,                     # ← 直接內嵌，不交給 COLLECT
    a.zipfiles,                     # ← 同上
    a.datas,                        # ← 同上
    name='LDA分析工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,                  # GUI 模式，不顯示終端視窗
    disable_windowed_traceback=False,
    argv_emulation=IS_MAC,          # Apple Event 處理，僅 macOS 需要
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_icon,
)

# ── macOS .app bundle ─────────────────────────────────────────────────────────
# COLLECT 已移除；BUNDLE 直接接收 exe，產出單一 .app，dist/ 下無多餘資料夾
if IS_MAC:
    app = BUNDLE(
        exe,                        # ← 直接包 exe，不經 COLLECT
        name='LDA分析工具.app',
        icon=_icon,
        bundle_identifier='com.lda.gui',
        info_plist={
            'NSHighResolutionCapable': True,
            'CFBundleShortVersionString': '1.0.0',
        },
    )
