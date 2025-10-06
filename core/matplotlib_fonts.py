import matplotlib
import matplotlib.pyplot as plt
import os


def set_korean_font(font_name: str = None):
    """
    Try to configure matplotlib to use a Korean-capable font.

    Strategy:
    - If `font_name` provided and available, use it.
    - Otherwise try common macOS / Noto fonts.
    """
    candidates = [font_name] if font_name else []
    # Common macOS and cross-platform fonts that support Korean
    candidates += ['AppleGothic', 'Apple SD Gothic Neo', 'Noto Sans KR', 'NanumGothic', 'Malgun Gothic']

    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}

    for cand in candidates:
        if cand and cand in available:
            matplotlib.rcParams['font.family'] = cand
            return cand

    # If none found, attempt to use the first available font that has 'Noto' or 'Nanum'
    for f in available:
        if 'Noto' in f or 'Nanum' in f:
            matplotlib.rcParams['font.family'] = f
            return f

    # fallback: leave default (warnings may persist)
    return None
