#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_image.py — 图片结构化分析：让纯文本模型"看懂"图片

功能:
  1. OCR 文字提取（Windows 自带 OCR 引擎，双遍识别取并集）
  2. 形状识别（矩形/圆形/线段/箭头/多边形）与位置、大小
  3. 颜色分析（整体主色 + 每个区域的颜色）
  4. 布局结构描述（顶栏/侧边栏/内容分区）
  5. 彩色字符画（形状与颜色一体呈现，纯文本模型可读懂色码）

用法:
  python analyze_image.py <图片路径> [--width 72] [--plain] [--no-ascii] [--out 文件] [--debug]
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
OCR_PS1 = SCRIPT_DIR / "ocr.ps1"
MIN_REGION_FRAC = 0.004    # 区域最小面积占比（忽略过小噪点）
MAX_ANALYSIS_DIM = 1000    # 形状分析用最大边长（控制内存与速度）
MAX_OCR_DIM = 3000         # OCR 前最大边长（超大图先缩放再识别）
CHARS = " .:-=+*#%@"       # 亮度 → 字符（暗→密）

COLOR_TABLE = [
    ("黑色", (0, 0, 0)), ("深灰", (64, 64, 64)), ("灰色", (128, 128, 128)),
    ("浅灰", (196, 196, 196)), ("白色", (255, 255, 255)),
    ("深红", (150, 30, 30)), ("红色", (217, 70, 65)), ("粉色", (236, 150, 180)),
    ("橙色", (240, 140, 40)), ("金色", (200, 165, 60)), ("黄色", (245, 205, 40)),
    ("深绿", (30, 110, 50)), ("绿色", (60, 170, 80)), ("浅绿", (150, 220, 160)),
    ("青色", (40, 180, 180)), ("深蓝", (31, 78, 140)), ("蓝色", (60, 110, 210)),
    ("浅蓝", (130, 180, 240)), ("紫色", (140, 70, 180)), ("棕色", (140, 95, 55)),
]


def hex_of(rgb):
    r, g, b = (int(round(v)) for v in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def rgb_of_hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def color_name(rgb):
    r, g, b = (float(v) for v in rgb)
    best, best_d = "未知", float("inf")
    for name, (cr, cg, cb) in COLOR_TABLE:
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_d:
            best, best_d = name, d
    return best


# ---------------- OCR ----------------

def _ocr_pass(img_path, scale):
    """按指定缩放比做一次 OCR，返回原图坐标的单词列表；失败返回 None"""
    out_json = Path(tempfile.gettempdir()) / f"_ocr_out_{os.getpid()}_{scale}.json"
    tmp_png = None
    ocr_path = str(img_path)
    if abs(scale - 1.0) > 0.01:
        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
            tmp_png = str(Path(tempfile.gettempdir()) / f"_ocr_{os.getpid()}_{scale}.png")
            im.save(tmp_png)
        ocr_path = tmp_png
    try:
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
               "-File", str(OCR_PS1), "-ImagePath", ocr_path, "-OutFile", str(out_json)]
        subprocess.run(cmd, capture_output=True, timeout=120)
        if not out_json.exists():
            return None
        data = json.loads(out_json.read_text(encoding="utf-8-sig"))
    finally:
        if tmp_png:
            try:
                os.remove(tmp_png)
            except OSError:
                pass
        try:
            os.remove(out_json)
        except OSError:
            pass
    words = data.get("words", [])
    for w in words:
        w["x"] = int(w["x"] / scale)
        w["y"] = int(w["y"] / scale)
        w["w"] = int(w["w"] / scale)
        w["h"] = int(w["h"] / scale)
    return words


def ocr_windows(img_path):
    """双遍 OCR（原尺寸 + 放大），取并集去重；0 字时增强对比度重试（暗色模式截图）"""
    with Image.open(img_path) as im:
        w0, h0 = im.size
    if max(w0, h0) > MAX_OCR_DIM:
        passes = [MAX_OCR_DIM / max(w0, h0)]
    elif max(w0, h0) < 600:
        passes = [1.0, 3.0]   # 小截图：原尺寸 + 3 倍放大
    elif max(w0, h0) < 1600:
        passes = [1.0, 2.0]   # 中图：原尺寸 + 2 倍放大互补
    else:
        passes = [1.0]

    all_words = []
    with ThreadPoolExecutor(max_workers=2) as ex:   # 双遍并行，节省一半时间
        for ws in ex.map(lambda s: _ocr_pass(img_path, s), passes):
            if ws:
                all_words.extend(ws)

    # 全部失败且对比度低（暗色模式等）→ 拉伸对比度后重试一次
    if not all_words:
        try:
            with Image.open(img_path) as im:
                arr = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.float32)
            lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
            if hi - lo > 20:
                arr = np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
                enh_path = str(Path(tempfile.gettempdir()) / f"_ocr_enh_{os.getpid()}.png")
                Image.fromarray(arr).save(enh_path)
                try:
                    # 增强后放大到 ~2000px 再识别（小图暗色文字需要同时处理对比度和字号）
                    ws = _ocr_pass(enh_path, min(2.5, 2000 / max(w0, h0)))
                    if ws:
                        all_words.extend(ws)
                finally:
                    try:
                        os.remove(enh_path)
                    except OSError:
                        pass
        except Exception:
            pass
    return all_words   # 去重在 ocr_all 统一做（跨引擎合并）


_RAPID = None
_RAPID_OK = None


def _rapid_available():
    global _RAPID_OK
    if _RAPID_OK is None:
        try:
            import rapidocr_onnxruntime  # noqa: F401
            _RAPID_OK = True
        except ImportError:
            _RAPID_OK = False
    return _RAPID_OK


def ocr_rapid(img_path):
    """RapidOCR（PaddleOCR 的 ONNX 轻量版）识别，返回 [{text,x,y,w,h}]（原图坐标）"""
    global _RAPID
    try:
        if _RAPID is None:
            from rapidocr_onnxruntime import RapidOCR
            _RAPID = RapidOCR()
        result, _ = _RAPID(str(img_path))
    except Exception:
        return None
    words = []
    for box, text, score in (result or []):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        words.append({"text": str(text), "x": int(min(xs)), "y": int(min(ys)),
                      "w": int(max(xs) - min(xs)), "h": int(max(ys) - min(ys))})
    return words


def _dedup_words(all_words):
    """单词级去重：中心落在已有单词内，或高度重叠且 x 接近（跨引擎/跨遍数合并）。
    先按位置排序保证确定性（并行引擎完成顺序不定，去重结果不能依赖它）"""
    all_words = sorted(all_words, key=lambda w: (w["y"], w["x"]))
    uniq = []
    for w in all_words:
        dup = False
        cx, cy = w["x"] + w["w"] / 2, w["y"] + w["h"] / 2
        for u in uniq:
            if u["x"] <= cx <= u["x"] + u["w"] and u["y"] <= cy <= u["y"] + u["h"]:
                dup = True
                break
            ov = max(0, min(w["y"] + w["h"], u["y"] + u["h"]) - max(w["y"], u["y"]))
            if ov > 0.6 * min(w["h"], u["h"]) and abs(w["x"] - u["x"]) < 0.5 * max(w["w"], u["w"]):
                dup = True
                break
        if not dup:
            uniq.append(w)
    return uniq


def ocr_all(img_path, mode):
    """按 --ocr 模式并行运行各 OCR 引擎，合并去重后返回单词列表。
    auto = Windows +（已安装 RapidOCR 时）RapidOCR 并行"""
    engines = []
    if mode in ("auto", "windows", "both"):
        engines.append(ocr_windows)
    if mode in ("rapid", "both") or (mode == "auto" and _rapid_available()):
        engines.append(ocr_rapid)
    if not engines:
        return []
    words = []
    with ThreadPoolExecutor(max_workers=len(engines)) as ex:
        futures = [ex.submit(fn, img_path) for fn in engines]
        for fut in as_completed(futures):
            try:
                ws = fut.result()
            except Exception:
                continue
            if ws:
                words.extend(ws)
    return _dedup_words(words)


def cjk(ch):
    return "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f"


def merge_words_to_lines(words):
    """把 OCR 单词按行合并成文本行（解决中文按字/短语拆分的问题）"""
    def overlap_v(a, b):
        return max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))

    lines = []  # 每行: {"words": [...], "x","y","w","h"}（增量维护包围盒）
    for w in sorted(words, key=lambda w: (w["y"], w["x"])):
        for ln in lines:
            if overlap_v(ln, w) > 0.6 * min(ln["h"], w["h"]):
                nx = min(ln["x"], w["x"])
                ny = min(ln["y"], w["y"])
                nw = max(ln["x"] + ln["w"], w["x"] + w["w"]) - nx
                nh = max(ln["y"] + ln["h"], w["y"] + w["h"]) - ny
                ln.update(x=nx, y=ny, w=nw, h=nh)
                ln["words"].append(w)
                break
        else:
            lines.append({"words": [w], "x": w["x"], "y": w["y"], "w": w["w"], "h": w["h"]})

    out = []
    for ln in lines:
        ws = sorted(ln["words"], key=lambda w: w["x"])
        parts, prev = [], None
        for w in ws:
            t = w["text"]
            if prev is not None:
                gap = w["x"] - (prev["x"] + prev["w"])
                # 相邻都为中文时直接拼接；否则按间隙决定是否加空格
                if gap > 4 and not (cjk(prev["text"][-1]) and cjk(t[0])):
                    parts.append(" ")
            parts.append(t)
            prev = w
        out.append({"text": "".join(parts), "x": ln["x"], "y": ln["y"], "w": ln["w"], "h": ln["h"]})
    return out


# ---------------- 形状/区域检测 ----------------

def classify_shape(cnt, w, h, fill, area):
    peri = cv2.arcLength(cnt, True)
    if peri <= 0 or w == 0 or h == 0:
        return "区域"
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    n = len(approx)
    if is_arrow(cnt, w, h):
        return "箭头"
    if n == 3 and fill >= 0.5:
        return "三角形"
    if n == 4:
        if fill >= 0.5:
            return "矩形"
        # 菱形 vs 描边矩形：角点贴 bbox 边缘 → 描边矩形；角点居中 → 菱形
        pts = approx.reshape(-1, 2)
        bx, by = int(pts[:, 0].min()), int(pts[:, 1].min())
        bw2, bh2 = int(pts[:, 0].max()) - bx, int(pts[:, 1].max()) - by
        near_edge = sum(1 for (px, py) in pts
                        if px <= bx + 2 or px >= bx + bw2 - 2 or py <= by + 2 or py >= by + bh2 - 2)
        if near_edge >= 3 and fill >= 0.12:
            return "矩形框(描边)"
        if fill >= 0.3:
            return "菱形/斜四边形"
    aspect = max(w, h) / max(1, min(w, h))
    if aspect >= 5 and fill < 0.5:
        return "线段/细条"
    circ = 4 * math.pi * area / (peri * peri)
    if n >= 6:
        if aspect >= 1.2 and fill >= 0.5:
            return "圆角矩形"      # 常规圆角卡片/按钮/胶囊（优先于圆形判定）
        if aspect < 0.6:
            return "线段/细条" if fill < 0.5 else "圆角矩形"
        if circ >= 0.75:
            return "圆形/椭圆"
        if fill >= 0.7:
            return "圆角矩形"
    if fill >= 0.55 and circ >= 0.6:
        return "圆形/椭圆"
    return "不规则形状"


def is_arrow(cnt, w, h):
    """启发式：细长区域 + 一端有深凹口 → 箭头"""
    try:
        if min(w, h) == 0 or max(w, h) / min(w, h) < 1.8:
            return False
        hull = cv2.convexHull(cnt, returnPoints=False)
        if hull is None or len(hull) < 3 or len(cnt) < 5:
            return False
        defects = cv2.convexityDefects(cnt, hull)
        if defects is None:
            return False
        deepest = 0.0
        for d in defects:
            d = np.asarray(d).reshape(-1)  # 兼容 OpenCV 4/5 的返回形状
            deepest = max(deepest, d[3] / 256.0)
        return deepest > min(w, h)
    except cv2.error:
        return False   # 自交轮廓等异常输入：convexityDefects 会报错，判为非箭头


def detect_regions(bgr, aW, aH):
    """颜色量化 + 连通域分析 → (区域列表, 调色板)；坐标基于分析图"""
    H, W = bgr.shape[:2]
    total = H * W
    min_area = MIN_REGION_FRAC * total
    pixels = bgr.reshape(-1, 3).astype(np.float32)

    # k-means 颜色量化（采样加速）
    n = min(30000, len(pixels))
    sample = pixels[np.random.default_rng(0).choice(len(pixels), n, replace=False)]
    # 暗图预处理：对比度拉伸后再聚类（深色图各色接近，直接聚类会糊成一团）。
    # 拉伸只用于分类；区域颜色和调色板仍用原始色（见下方 orig_color）
    dark_lo = dark_hi = None
    if float(pixels.mean()) < 110:
        dark_lo, dark_hi = np.percentile(sample, 2), np.percentile(sample, 98)
        if dark_hi - dark_lo > 20:
            sample = np.clip((sample - dark_lo) * 255.0 / (dark_hi - dark_lo), 0, 255).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(sample, 14, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    # 合并相近色（用 L1 距离：白色 #FEFEFE 与浅灰 #F5F5F5 应区分，L2 会误合并）
    merged = []
    for c in centers:
        if not any(np.abs(c - m).sum() < 20 for m in merged):
            merged.append(c)
    merged = np.array(merged, dtype=np.float32)

    # 全图像素归属到最近的量化色（分块，避免大图内存峰值）
    labels = np.empty(len(pixels), dtype=np.int32)
    for start in range(0, len(pixels), 100000):
        chunk = pixels[start:start + 100000]
        if dark_lo is not None:
            chunk = np.clip((chunk - dark_lo) * 255.0 / (dark_hi - dark_lo), 0, 255).astype(np.float32)
        dists = np.abs(chunk[:, None, :] - merged[None, :, :]).sum(axis=2)
        labels[start:start + len(chunk)] = dists.argmin(axis=1)
    labels = labels.reshape(H, W)

    # 暗图：聚类的"原始色" = 该聚类像素的原始值中位数
    orig_color = {}
    if dark_lo is not None:
        flat = labels.ravel()
        for ci in range(len(merged)):
            m = flat == ci
            if m.any():
                orig_color[ci] = np.median(pixels[m], axis=0)

    regions = []
    for ci, c in enumerate(merged):
        mask = (labels == ci).astype(np.uint8) * 255
        if mask.sum() < min_area:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # 真实像素数：用闭运算前的 label 精确计数（闭运算会膨胀掩膜，
            # 相邻色块互相重叠导致面积虚高）
            px_count = int(np.count_nonzero(labels[y:y + h, x:x + w] == ci))
            regions.append({
                "x": x, "y": y, "w": w, "h": h,
                "shape": classify_shape(cnt, w, h, px_count / max(1, w * h), px_count),
                "area": px_count, "fill": px_count / max(1, w * h),
            })

    # 背景识别：覆盖面积 >25% 且接触 ≥3 条边 → 背景色块，不进元素表（单独返回供主色统计）
    bg_regions = []
    kept_all, regions = regions, []
    for r in kept_all:
        if r["area"] > 0.25 * total and sum([r["x"] <= 2, r["y"] <= 2,
                                             r["x"] + r["w"] >= aW - 2, r["y"] + r["h"] >= aH - 2]) >= 3:
            bg_regions.append(r)
        else:
            regions.append(r)

    # 区域真实主色：bbox 内出现最多的量化色（比聚类中心更准，排除文字/边框干扰）
    for r in regions:
        sub = labels[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]].ravel()
        vals, counts = np.unique(sub, return_counts=True)
        main = vals[counts.argmax()]
        r["color"] = tuple(int(v) for v in orig_color.get(main, merged[main]))

    # 重叠去重：只有"小而实心"的块才吞并完全被它盖住的小区域
    regions.sort(key=lambda r: r["area"], reverse=True)
    kept = []
    for r in regions:
        # 抗锯齿/边框碎片：真实像素少且实心度极低 → 丢弃
        if r["area"] < min_area and r["fill"] < 0.2:
            continue
        if any(_swallowed(r, k, total) for k in kept):
            continue
        # 近重复区域（同一色块被切成多个连通件）→ 按包围盒 IoU 去重
        if any(_bbox_iou(r, k) > 0.7 for k in kept):
            continue
        kept.append(r)
    kept.sort(key=lambda r: (r["y"], r["x"]))

    for r in kept:
        r["color_hex"] = hex_of(r["color"])
        r["color_name"] = color_name(r["color"])
    for r in bg_regions:   # 背景色块同样需要颜色（供主色统计）
        sub = labels[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]].ravel()
        vals, counts = np.unique(sub, return_counts=True)
        main = vals[counts.argmax()]
        r["color"] = tuple(int(v) for v in orig_color.get(main, merged[main]))
        r["color_hex"] = hex_of(r["color"])
        r["color_name"] = color_name(r["color"])
    palette = [tuple(int(v) for v in orig_color.get(ci, merged[ci])) for ci in range(len(merged))]
    return kept, palette, bg_regions


def _overlap_area(a, b):
    """两个矩形（x,y,w,h）的重叠面积；不相交时钳制为 0（防止负负得正的幻影重叠）"""
    ox = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    oy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    return max(0, ox) * max(0, oy)


def _bbox_iou(a, b):
    """包围盒 IoU（用于近重复区域去重；badge 之类的小区域和容器 IoU 极小，不会误删）"""
    ox = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    oy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    inter = max(0, ox) * max(0, oy)
    union = max(1, a["w"] * a["h"] + b["w"] * b["h"] - inter)
    return inter / union


def _swallowed(r, k, total):
    """小区域 r 是否被实心块 k 吞并：k 必须小而实心（实心大背景里的
    徽章/按钮是真实元素，不能被吞；小实心块内的噪点才该被吞）"""
    if k["fill"] <= 0.92 or k["area"] >= 0.3 * total:
        return False
    return _overlap_area(r, k) > 0.8 * r["area"]


def _direction(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) < 3:
        return "↓" if dy > 0 else "↑"
    if abs(dy) < 3:
        return "→" if dx > 0 else "←"
    return ("↘" if dx > 0 else "↙") if dy > 0 else ("↗" if dx > 0 else "↖")


def _darkest(gray, x, y, w, h):
    sub = gray[max(0, y):y + h + 1, max(0, x):x + w + 1]
    v = int(sub.min())
    return (v, v, v)


def _tip_is_arrowhead(gray, x1, y1, x2, y2):
    """判断线段端点 (x2,y2) 处是否有箭头头部：
    沿端点做垂直扫描线采样宽度剖面——箭头头部的剖面是一个有界凸起
    （峰值≥10px，且两侧 15px 内都回落到≤5px）；实心块、边框、普通线不会这样。
    杆线常止步于头部基部前 10-20px，因此采样范围取 ±33px"""
    ang = math.atan2(y2 - y1, x2 - x1)
    ux, uy = math.cos(ang), math.sin(ang)
    px, py = -math.sin(ang), math.cos(ang)
    h, w = gray.shape

    def width_at(d):
        bx, by = int(x2 + ux * d), int(y2 + uy * d)
        cnt = 0
        for s in range(-25, 26):
            sx, sy = int(bx + px * s), int(by + py * s)
            if 0 <= sx < w and 0 <= sy < h and gray[sy, sx] < 110:
                cnt += 1
        return cnt

    ws = [width_at(d) for d in range(-33, 34, 3)]
    peak = max(ws)
    if peak < 10:   # 太弱的凸起可能是线杆截交/文字笔画，不是箭头头部
        return False
    dstar = ws.index(peak) * 3 - 33

    def dropped(step):
        for k in range(1, 6):   # 15px 内
            idx = (dstar + step * k * 3 + 33) // 3
            if 0 <= idx < len(ws) and ws[idx] <= 5:
                return True
        return False

    return dropped(1) and dropped(-1)


def _on_region_border(bx, by, bw, bh, r):
    """线段（bbox 形式）是否贴着某个区域的边缘（节点/色块的边框线）"""
    if bh <= bw:   # 横向线
        for edge_y in (r["y"], r["y"] + r["h"]):
            if abs(by - edge_y) < 4:
                span = min(bx + bw, r["x"] + r["w"]) - max(bx, r["x"])
                return span >= 0.6 * max(r["w"], bw)
    else:          # 纵向线
        for edge_x in (r["x"], r["x"] + r["w"]):
            if abs(bx - edge_x) < 4:
                span = min(by + bh, r["y"] + r["h"]) - max(by, r["y"])
                return span >= 0.6 * max(r["h"], bh)
    return False


def _collinear(e, bx, by, bw, bh):
    """方向一致且相距近 → 属于同一条线（霍夫常把一条线拆成多段，需合并）。
    横/竖方向必须一致，交叉线（如表格外框的横边和竖边）不能合并"""
    if e["w"] < 6 and e["h"] < 6:
        return False
    if (e["w"] >= e["h"]) != (bw >= bh):
        return False
    if e["w"] >= e["h"]:   # 横向
        if abs(by - e["y"]) > 6:
            return False
        return not (bx > e["x"] + e["w"] + 20 or bx + bw < e["x"] - 20)
    else:                  # 纵向
        if abs(bx - e["x"]) > 6:
            return False
        return not (by > e["y"] + e["h"] + 20 or by + bh < e["y"] - 20)


def _line_is_solid(gray, x1, y1, x2, y2):
    """沿线采样：每个采样点在垂直方向 ±2px 窗口取极值（容忍霍夫线落在
    描边边缘上 1-2px 的偏移），与背景（±6~10px）差异 ≥25 的比例 ≥60% → 真实线条。
    文字笔画链在字符间隙处取不到深色，会被过滤"""
    ang = math.atan2(y2 - y1, x2 - x1)
    px, py = -math.sin(ang), math.cos(ang)
    h, w = gray.shape
    n = 20
    solid = 0
    for i in range(n + 1):
        t = i / n
        x, y = int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t)
        if not (0 <= x < w and 0 <= y < h):
            continue
        fg = []
        for s in range(-2, 3):
            sx, sy = int(x + px * s), int(y + py * s)
            if 0 <= sx < w and 0 <= sy < h:
                fg.append(gray[sy, sx])
        bg = []
        for s in (6, 7, 8, 9, 10):
            for sign in (1, -1):
                sx, sy = int(x + px * s * sign), int(y + py * s * sign)
                if 0 <= sx < w and 0 <= sy < h:
                    bg.append(gray[sy, sx])
        if not fg or not bg:
            continue
        med_bg = np.median(bg)
        if min(fg) < med_bg - 25:            # 窗口内有深色（细描边也能抓到）
            line_val = min(fg)
        elif max(fg) > med_bg + 25:          # 浅色线（暗色主题）
            line_val = max(fg)
        else:
            line_val = int(med_bg)
        if abs(int(line_val) - int(med_bg)) > 25:
            solid += 1
    return solid / (n + 1) >= 0.6


def _is_separator(gray, coord, a, b, axis, half=12, dark_frac_thr=0.15):
    """表格分隔线判别：线两侧 ±half 带内几乎没有与背景差异明显的像素 → 分隔线。
    排除线自身像素（线本身总是深色）；背景自适应（浅底/暗底都适用）。
    文字行两侧全是笔画，会被排除"""
    if axis == "h":
        band = gray[max(0, coord - half):coord, a:b]
        band2 = gray[coord + 1:coord + half + 1, a:b]
    else:
        band = gray[a:b, max(0, coord - half):coord]
        band2 = gray[a:b, coord + 1:coord + half + 1]
    total = band.size + band2.size
    if total == 0:
        return False
    both = np.concatenate([band.ravel(), band2.ravel()])
    med = float(np.median(both))
    diff = int((np.abs(band.astype(np.int16) - med) > 30).sum()) \
        + int((np.abs(band2.astype(np.int16) - med) > 30).sum())
    return diff / total < dark_frac_thr


def detect_lines(gray, regions, ocr_boxes, aW, aH):
    """霍夫线段检测：细连线/分隔线/箭头（粗箭头由 detect_regions 处理）。
    流程：过滤边框与文字笔画 → 合并共线碎片 → 统一做箭头分类。
    ocr_boxes: 文字行区域（分析图坐标，已外扩），与文字行重叠的线是文字笔画"""
    min_len = int(min(aW, aH) * 0.06)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=min_len, maxLineGap=8)
    merged = []
    if lines is None:
        return []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):   # 兼容 OpenCV 4/5 返回形状
        # 一律转成 bbox 形式（min/max），杜绝端点顺序/斜向线的负宽高
        bx, by = min(x1, x2), min(y1, y2)
        bw, bh = abs(x2 - x1), abs(y2 - y1)
        if math.hypot(bw, bh) < min_len:
            continue
        dir_str = _direction(x1, y1, x2, y2)   # 方向用原始端点（保持箭头指向语义）
        if any(_on_region_border(bx, by, bw, bh, r) for r in regions):
            continue   # 节点/色块的边框线
        # 区域内部的线：只有结构线（分隔线/表格线）才有意义，文字笔画会被
        # 霍夫串成短"线"，按跨度过滤掉；内容容器内同样只保留横贯容器的线
        mx, my = bx + bw / 2, by + bh / 2
        noise = False
        for r in regions:
            if r["x"] <= mx <= r["x"] + r["w"] and r["y"] <= my <= r["y"] + r["h"]:
                if r["area"] > 0.2 * aW * aH:
                    if not (bw >= 0.5 * r["w"] or bh >= 0.5 * r["h"]):
                        noise = True
                else:
                    if bw < 0.6 * r["w"] and bh < 0.6 * r["h"]:
                        noise = True
                break
        if noise:
            continue
        if not _line_is_solid(gray, x1, y1, x2, y2):
            continue   # 文字笔画链等断续暗块
        # 中心落在某条 OCR 文字行内 → 是文字笔画而不是真实线条
        if any(bx + bw / 2 >= ox and bx + bw / 2 <= ox + ow
               and by + bh / 2 >= oy and by + bh / 2 <= oy + oh
               for ox, oy, ow, oh in ocr_boxes):
            continue
        # 合并共线碎片（霍夫常把一条线拆成多段）
        for e in merged:
            if _collinear(e, bx, by, bw, bh):
                e["x"] = min(e["x"], bx)
                e["y"] = min(e["y"], by)
                e["w"] = max(e["x"] + e["w"], bx + bw) - e["x"]
                e["h"] = max(e["y"] + e["h"], by + bh) - e["y"]
                break
        else:
            merged.append({"x": bx, "y": by, "w": bw, "h": bh, "dir": dir_str})

    elems = []
    for e in merged:
        is_arrow = (_tip_is_arrowhead(gray, e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"])
                    or _tip_is_arrowhead(gray, e["x"] + e["w"], e["y"] + e["h"], e["x"], e["y"]))
        kind = "箭头" if is_arrow else "线段"
        elems.append({"x": e["x"], "y": e["y"], "w": e["w"], "h": e["h"],
                      "shape": f"{kind}({e['dir']})",
                      "area": int(math.hypot(e["w"], e["h"])), "fill": 0.5,
                      "color": _darkest(gray, e["x"], e["y"], e["w"], e["h"])})
    return elems


# ---------------- 表格识别 ----------------

def _cluster_lines(lines, tol=6):
    """聚类相近的平行线（按主坐标，容差 tol px）：返回 [(主坐标, 起点, 终点)]。
    霍夫会把同一条线拆成多段，聚类后每簇代表一条表格线"""
    groups = []
    for coord, a, b in sorted(lines, key=lambda t: t[0]):
        for g in groups:
            if abs(coord - g[0]) <= tol:
                g[0] = (g[0] * g[3] + coord) / (g[3] + 1)   # 增量平均
                g[1] = min(g[1], a)
                g[2] = max(g[2], b)
                g[3] += 1
                break
        else:
            groups.append([coord, a, b, 1])
    return [(g[0], g[1], g[2]) for g in groups]


def detect_tables(gray, words, aW, aH, min_len=25):
    """表格网格识别：水平/垂直线聚类 → 网格校验 → 单元格文字 → markdown 表格。
    words: 分析图坐标的 OCR 单词（含 x,y,w,h,text）。返回 markdown 表格字符串列表"""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=min_len, maxLineGap=6)
    if lines is None:
        return []
    hs, vs = [], []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        if abs(y2 - y1) <= abs(x2 - x1):          # 横向线
            if abs(x2 - x1) >= min_len and _line_is_solid(gray, x1, y1, x2, y2) \
                    and _is_separator(gray, (y1 + y2) // 2, min(x1, x2), max(x1, x2), "h"):
                hs.append((min(y1, y2), min(x1, x2), max(x1, x2)))
        elif abs(y2 - y1) >= min_len and _line_is_solid(gray, x1, y1, x2, y2) \
                and _is_separator(gray, (x1 + x2) // 2, min(y1, y2), max(y1, y2), "v", half=8):
            vs.append((min(x1, x2), min(y1, y2), max(y1, y2)))

    rows = _cluster_lines(hs)     # 行边界 [(y, x1, x2)]
    cols = _cluster_lines(vs)     # 列边界 [(x, y1, y2)]
    if len(rows) < 2:
        return []
    # 表格线横贯表格宽度：保留跨度 ≥ 最长线 50% 的线（文字碎片线跨度很短）
    rows = _span_filter(rows)
    cols = _span_filter(cols)
    if len(rows) < 2:
        return []
    # 行边界间距应大致均匀（表格行）：丢弃孤立行（如底部提示文字产生的线）
    rows = _regular_rows(rows)
    if len(rows) < 2:
        return []
    # 行线应互相横向对齐（同一张表的行线覆盖几乎相同的 x 范围；
    # 流程图节点边框各在各的 x 范围，会在此被排除）
    ix1, ix2 = max(b[1] for b in rows), min(b[2] for b in rows)
    ux1, ux2 = min(b[1] for b in rows), max(b[2] for b in rows)
    if ux2 - ux1 > 40 and ix2 - ix1 < 0.85 * (ux2 - ux1):
        return []
    if len(cols) >= 2:
        result = _grid_table_md(rows, cols, words)
        if result:
            return result
    return _infer_table_no_lines(words, rows)   # 无竖线表格 / 网格校验失败：列从文字推断


def _span_filter(lines, ratio=0.5):
    """保留跨度 ≥ 最长线 ratio 倍的线（表格线横贯表格，文字碎片线很短）"""
    if not lines:
        return lines
    max_span = max(b[2] - b[1] for b in lines)
    return [b for b in lines if b[2] - b[1] >= max_span * ratio]


def _regular_rows(rows, max_ratio=2.5):
    """保留间距均匀的连续行边界段（表格行高一致；间距异常大的孤立行是干扰）"""
    rs = sorted(rows, key=lambda b: b[0])
    if len(rs) <= 2:
        return rs
    gaps = [rs[i + 1][0] - rs[i][0] for i in range(len(rs) - 1)]
    med = sorted(gaps)[len(gaps) // 2]
    if med <= 0:
        return rs
    best, cur = [rs[0]], [rs[0]]
    for i in range(len(gaps)):
        if gaps[i] <= max_ratio * med:
            cur.append(rs[i + 1])
        else:
            if len(cur) > len(best):
                best = cur
            cur = [rs[i + 1]]
    if len(cur) > len(best):
        best = cur
    return best


def _grid_table_md(rows, cols, words):
    """网格表格：行/列线围成单元格，单词中心落入单元格 → markdown；校验失败返回 []"""
    tx1 = min(b[1] for b in cols)
    tx2 = max(b[2] for b in cols)
    ty1 = min(b[0] for b in rows)
    ty2 = max(b[0] for b in rows)
    width, height = tx2 - tx1, ty2 - ty1
    if width < 40 or height < 40:
        return []

    # 校验：行列线的跨度应覆盖表格的大部分范围（防止图表坐标轴等误判）
    cover_h = sum(max(0, min(b[2], tx2) - max(b[1], tx1)) for b in rows)
    cover_v = sum(max(0, min(b[2], ty2) - max(b[1], ty1)) for b in cols)
    if cover_h < 0.5 * width * len(rows) or cover_v < 0.5 * height * len(cols):
        return []

    # 单元格文字：单词中心落入对应单元格
    grid = [["" for _ in range(len(cols) - 1)] for _ in range(len(rows) - 1)]
    for wd in words:
        cx, cy = wd["x"] + wd["w"] / 2, wd["y"] + wd["h"] / 2
        for i in range(len(rows) - 1):
            if rows[i][0] <= cy <= rows[i + 1][0]:
                for j in range(len(cols) - 1):
                    if cols[j][0] <= cx <= cols[j + 1][0]:
                        grid[i][j] = (grid[i][j] + " " + wd["text"]).strip()
                break

    # 丢弃全空的行/列（文字笔画产生的幻影网格线会生成空行）
    grid = [row for row in grid if any(c for c in row)]
    if not grid:
        return []
    cols_keep = [j for j in range(len(grid[0]))
                 if any(grid[i][j] for i in range(len(grid)))]
    grid = [[grid[i][j] for j in cols_keep] for i in range(len(grid))]
    if len(grid) < 2 or len(grid[0]) < 2:   # 列过滤后必须仍是 ≥2×2
        return []

    md = []
    for i, row in enumerate(grid):
        md.append("| " + " | ".join(c or "—" for c in row) + " |")
        if i == 0:
            md.append("|" + "---|" * len(row))
    return ["\n".join(md)]


def _infer_table_no_lines(words, rows):
    """无竖线表格：只有横分隔线，列边界从文字中心 x 聚类推断"""
    ty1, ty2 = rows[0][0], rows[-1][0]
    ws = [w for w in words if ty1 <= w["y"] + w["h"] / 2 <= ty2]
    if len(ws) < 4:
        return []
    widths = sorted(w["w"] for w in ws)
    med_w = widths[len(widths) // 2]
    tol = max(12, med_w * 1.2)   # 单元格内字符间距≈字宽，同一格的字要能聚成一列

    # 文字中心 x 聚类 → 列
    clusters = []
    for w in sorted(ws, key=lambda w: w["x"] + w["w"] / 2):
        c = w["x"] + w["w"] / 2
        for cl in clusters:
            if abs(c - cl[0]) <= tol:
                cl[0] = (cl[0] * cl[2] + c) / (cl[2] + 1)
                cl[1].append(w)
                cl[2] += 1
                break
        else:
            clusters.append([c, [w], 1])
    clusters.sort(key=lambda cl: cl[0])
    merged = []
    for cl in clusters:
        if merged and cl[0] - merged[-1][0] < med_w:   # 过近的列合并（单元格内多词）
            m = merged[-1]
            m[0] = (m[0] * m[2] + cl[0] * cl[2]) / (m[2] + cl[2])
            m[1] += cl[1]
            m[2] += cl[2]
        else:
            merged.append(cl)
    merged = [cl for cl in merged if cl[2] >= 2]        # 每列至少出现 2 词（跨行）
    if len(merged) < 2:
        return []
    # 列中心必须在行线跨度内（行线即表格宽度；跨度外的文字是侧边栏等干扰）
    tx1, tx2 = min(b[1] for b in rows), max(b[2] for b in rows)
    merged = [cl for cl in merged if tx1 - 10 <= cl[0] <= tx2 + 10]
    if len(merged) < 2:
        return []

    # 单元格填充
    rows_sorted = sorted(rows, key=lambda b: b[0])
    n_rows = len(rows_sorted) - 1
    grid = [["" for _ in range(len(merged))] for _ in range(n_rows)]
    for j, cl in enumerate(merged):
        for w in cl[1]:
            cy = w["y"] + w["h"] / 2
            for i in range(n_rows):
                if rows_sorted[i][0] <= cy <= rows_sorted[i + 1][0]:
                    grid[i][j] = (grid[i][j] + " " + w["text"]).strip()
                    break

    grid = [row for row in grid if any(c for c in row)]
    if len(grid) < 2:
        return []
    md = []
    for i, row in enumerate(grid):
        md.append("| " + " | ".join(c or "—" for c in row) + " |")
        if i == 0:
            md.append("|" + "---|" * len(row))
    return ["\n".join(md)]


def classify_controls(regions, aW, aH):
    """控件语义标注（启发式）：按钮/输入框/开关/徽章/指示点，写进 shape 字段。
    需在 analyze_layout 之前调用（pct 坐标在此自行计算）"""
    for r in regions:
        if r["shape"] not in ("矩形", "圆角矩形", "矩形框(描边)", "圆形/椭圆"):
            continue
        w, h = r["w"] / aW * 100, r["h"] / aH * 100
        if w <= 0 or h <= 0:
            continue
        aspect, area_pct, fill = w / h, w * h, r["fill"]
        shape = r["shape"]
        if shape == "圆形/椭圆" and aspect < 1.3 and fill > 0.5 and w < 6 and h < 6:
            r["shape"] = "指示点/圆形按钮"
        elif shape in ("矩形", "圆角矩形", "矩形框(描边)"):
            near_white = (r["color"][0] > 210 and r["color"][1] > 210 and r["color"][2] > 210)
            if aspect >= 2.5 and 8 <= w <= 70 and (fill < 0.4 or (fill > 0.6 and near_white)):
                r["shape"] = f"输入框({shape})"   # 细长 + 描边/浅底 → 输入框优先于按钮
            elif 200 <= area_pct <= 2000 and fill >= 0.5 and near_white:
                r["shape"] = f"卡片({shape})"      # 近白大块（2%-20% 面积）→ 内容卡片
            elif 1.2 <= aspect <= 8 and fill > 0.6 and 1.5 <= w <= 90 and 1.5 <= h <= 15 and not near_white:
                if r.get("text", "") or aspect >= 3.5:
                    r["shape"] = f"按钮({shape})"
                elif 1.4 <= aspect <= 3 and w <= 30 and area_pct < 200:
                    r["shape"] = f"徽章/开关({shape})"


# ---------------- 图标识别 ----------------

def _icon_templates():
    """运行时生成线性图标模板（48×48 边缘图，图标充满画布以对齐候选尺寸），
    每种图标 1-2 个风格变体（细线/粗线），免打包资源"""
    import math
    size = 48
    icons = {}

    def add(name, draw_fn, width=4):
        img = Image.new("L", (size, size), 255)
        d = ImageDraw.Draw(img)
        draw_fn(d, width)
        icons.setdefault(name, []).append(cv2.Canny(np.array(img), 100, 200))

    def line_style(d, width):
        return width
    _ = line_style  # 占位避免误用

    # 搜索（放大镜：大圆 + 斜柄）
    add("搜索", lambda d, w: (d.ellipse([8, 3, 40, 35], outline=0, width=w),
                              d.line([36, 32, 45, 43], fill=0, width=w + 1)), 4)
    # 设置（齿轮简化：圆 + 十字齿 + 中心孔，与时钟的"内指针"明显区分）
    def gear(d, w):
        d.ellipse([9, 9, 39, 39], outline=0, width=w)
        d.line([24, 4, 24, 44], fill=0, width=w)     # 上下齿（贯穿圆外）
        d.line([4, 24, 44, 24], fill=0, width=w)     # 左右齿
        d.ellipse([19, 19, 29, 29], outline=0, width=w + 1)
    add("设置", gear, 4)
    # 菜单（三横线）
    add("菜单", lambda d, w: (d.line([8, 14, 40, 14], fill=0, width=w),
                              d.line([8, 24, 40, 24], fill=0, width=w),
                              d.line([8, 34, 40, 34], fill=0, width=w)), 4)
    # 关闭（X）
    add("关闭", lambda d, w: (d.line([13, 13, 35, 35], fill=0, width=w),
                              d.line([35, 13, 13, 35], fill=0, width=w)), 4)
    # 返回（左箭头）
    add("返回", lambda d, w: (d.line([40, 24, 12, 24], fill=0, width=w),
                              d.line([12, 24, 23, 13], fill=0, width=w),
                              d.line([12, 24, 23, 35], fill=0, width=w)), 4)
    # 加号
    add("加号", lambda d, w: (d.line([24, 10, 24, 38], fill=0, width=w),
                              d.line([10, 24, 38, 24], fill=0, width=w)), 4)
    # 扫码/相机（机身圆角矩形 + 顶部突起 + 镜头）
    def camera(d, w):
        d.rounded_rectangle([6, 18, 42, 42], radius=4, outline=0, width=w)
        d.line([16, 14, 20, 10], fill=0, width=w)
        d.line([20, 10, 28, 10], fill=0, width=w)
        d.line([28, 10, 32, 14], fill=0, width=w)
        d.ellipse([17, 23, 31, 37], outline=0, width=w)
    add("扫码/相机", camera, 4)
    # 分享（三节点 + 连线）
    def share(d, w):
        for cx, cy in ((13, 14), (35, 24), (15, 34)):
            d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=0, width=w)
        d.line([17, 17, 31, 22], fill=0, width=w)
        d.line([17, 31, 31, 26], fill=0, width=w)
    add("分享", share, 4)
    # 收藏（五角星）
    def star(d, w):
        pts = []
        for i in range(10):
            r = 21 if i % 2 == 0 else 9
            a = math.radians(-90 + i * 36)
            pts.append((24 + r * math.cos(a), 24 + r * math.sin(a)))
        d.polygon(pts, outline=0, fill=None)
        for i in range(10):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % 10]
            d.line([x1, y1, x2, y2], fill=0, width=w)
    add("收藏", star, 4)
    # 点赞（心形：两圆 + 三角）
    def heart(d, w):
        d.ellipse([6, 8, 24, 26], outline=0, width=w)
        d.ellipse([24, 8, 42, 26], outline=0, width=w)
        d.line([9, 17, 24, 38], fill=0, width=w)
        d.line([39, 17, 24, 38], fill=0, width=w)
    add("点赞/喜欢", heart, 4)
    # 下载（下箭头 + 托盘线）
    def download(d, w):
        d.line([24, 8, 24, 32], fill=0, width=w)
        d.line([24, 32, 14, 22], fill=0, width=w)
        d.line([24, 32, 34, 22], fill=0, width=w)
        d.line([8, 40, 40, 40], fill=0, width=w)
    add("下载", download, 4)
    # 更多（三点横排）
    add("更多", lambda d, w: (d.ellipse([8, 20, 16, 28], fill=0),
                              d.ellipse([20, 20, 28, 28], fill=0),
                              d.ellipse([32, 20, 40, 28], fill=0)), 4)
    # 播放（右向三角形）
    def play(d, w):
        d.polygon([(14, 8), (40, 24), (14, 40)], outline=0, fill=None)
        d.line([14, 8, 40, 24], fill=0, width=w)
        d.line([40, 24, 14, 40], fill=0, width=w)
        d.line([14, 8, 14, 40], fill=0, width=w)
    add("播放", play, 4)
    # 用户/头像（圆头 + 肩弧）
    def user(d, w):
        d.ellipse([16, 6, 32, 22], outline=0, width=w)
        d.arc([6, 24, 42, 54], start=180, end=360, fill=0, width=w)
    add("用户", user, 4)
    # 时钟/历史（圆 + 中心点 + 长短指针；指针在圆内，与齿轮/放大镜区分）
    def clock(d, w):
        d.ellipse([6, 6, 42, 42], outline=0, width=w)
        d.ellipse([22, 22, 26, 26], fill=0)
        d.line([24, 24, 38, 24], fill=0, width=w)    # 分针（指 3 点，到圆边）
        d.line([24, 24, 24, 12], fill=0, width=w)    # 时针（指 12 点）
    add("时钟", clock, 4)
    # 删除（垃圾桶）
    def trash(d, w):
        d.line([12, 14, 36, 14], fill=0, width=w)
        d.line([17, 14, 17, 9], fill=0, width=w)
        d.line([17, 9, 31, 9], fill=0, width=w)
        d.line([31, 9, 31, 14], fill=0, width=w)
        d.line([15, 14, 17, 40], fill=0, width=w)
        d.line([33, 14, 31, 40], fill=0, width=w)
        d.line([17, 40, 31, 40], fill=0, width=w)
    add("删除", trash, 4)
    return icons


_ICON_TEMPLATES = None


def _refine_circle_icon(gray, x, y, w, h, name):
    """圆类图标后验区分：按 45° 扇区统计深色像素的最大半径——
    放大镜的柄使某一侧扇区半径突出、对角出现缺口（极差大）；
    时钟/齿轮的圆环对称均匀（极差小）。极差大 → 放大镜（搜索）"""
    if name not in ("搜索", "设置", "时钟"):
        return name
    ys, xs = np.nonzero(gray[y:y + h, x:x + w] < 150)
    if len(xs) < 20:
        return name
    cx, cy = w / 2, h / 2
    min_wh = max(1, min(w, h))
    sector_max = {}
    for px, py in zip(xs, ys):
        ang = int((np.degrees(np.arctan2(py - cy, px - cx)) + 360) % 360)
        sec = ang // 45
        r = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 / min_wh
        sector_max[sec] = max(sector_max.get(sec, 0), r)
    if sector_max and (max(sector_max.values()) - min(sector_max.values())) > 0.25:
        return "搜索"
    return name


def detect_icons(gray, aW, aH, word_boxes=None, regions=None):
    """小图标识别：深色小块（暗底图时含浅色块）→ 边缘模板匹配。
    word_boxes: 分析图坐标的 OCR 单词框，重叠的候选是文字笔画不是图标。
    regions: 已检测区域，重叠的小区域（徽章/按钮等）不再重复识别。
    返回区域风格的 dict 列表（shape="图标:名称"）"""
    global _ICON_TEMPLATES
    if _ICON_TEMPLATES is None:
        _ICON_TEMPLATES = _icon_templates()
    small_regions = [r for r in (regions or [])
                     if r["area"] < 0.15 * aW * aH]   # 大容器不参与排除
    _, dark = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    mask = dark
    if float(gray.mean()) < 110:          # 暗底图：图标多为浅色
        _, bright = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(dark, bright)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elems = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if not (12 <= max(w, h) <= 72 and 0.6 <= w / max(1, h) <= 1.8):
            continue
        if not (60 <= cv2.contourArea(cnt) <= 4000):
            continue
        cand = {"x": x, "y": y, "w": w, "h": h}
        if word_boxes:
            # 与文字框的总重叠 ≥ 候选面积 40% → 是文字笔画不是图标。
            # 只排除多字符词：单字符"词"（"三""×""+"）常是图标被 OCR 误读的产物
            total = sum(_overlap_area(cand, {"x": bx, "y": by, "w": bw, "h": bh})
                        for bx, by, bw, bh, wt in word_boxes if len(wt) >= 2)
            if total >= 0.4 * (w * h):
                continue
        if small_regions:
            # 实心小块（徽章/按钮/开关等）已被识别为区域 → 不重复识别；
            # 线条类图标（fill 低）仍保留
            cand_fill = cv2.contourArea(cnt) / max(1, w * h)
            if cand_fill >= 0.5 and any(_overlap_area(cand, r) >= 0.4 * (w * h) for r in small_regions):
                continue
        patch = cv2.resize(gray[y:y + h, x:x + w], (48, 48), interpolation=cv2.INTER_AREA)
        # 等比缩放 + 居中补白到 48×48（直接拉伸会把圆压成椭圆，模板匹配必败）
        scale = 40 / max(w, h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(gray[y:y + h, x:x + w], (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((48, 48), 255, np.uint8)
        ox, oy = (48 - nw) // 2, (48 - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = small
        edges = cv2.Canny(canvas, 100, 200)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))   # 容忍笔画粗细/位置微差
        best, best_s = None, 0.0
        for name, variants in _ICON_TEMPLATES.items():
            for tpl in variants:
                for sc in (0.75, 0.9, 1.0):
                    t = cv2.resize(tpl, (int(48 * sc), int(48 * sc)), interpolation=cv2.INTER_AREA)
                    t = cv2.dilate(t, np.ones((3, 3), np.uint8))
                    s = cv2.matchTemplate(edges, t, cv2.TM_CCOEFF_NORMED)[0][0]
                    if s > best_s:
                        best, best_s = name, s
        if best and best_s > 0.5:
            best = _refine_circle_icon(gray, x, y, w, h, best)   # 放大镜/时钟/设置后验区分
            elems.append({"x": x, "y": y, "w": w, "h": h, "shape": f"图标:{best}",
                          "area": int(cv2.contourArea(cnt)), "fill": 0.3,
                          "color": (80, 80, 80)})
    return elems


# ---------------- 布局分析 ----------------

def analyze_layout(regions, lines, aW, aH, ow, oh):
    """计算百分比坐标、挂接文字、归纳布局结构"""
    sx, sy = aW / ow, aH / oh
    total = aW * aH
    for r in regions:
        r["px"] = r["x"] / aW * 100
        r["py"] = r["y"] / aH * 100
        r["p_w"] = r["w"] / aW * 100
        r["p_h"] = r["h"] / aH * 100
        r["text"] = ""

    # OCR 行（原图坐标 → 分析图坐标）挂到覆盖它的区域上；
    # 多个区域都能盖住时选包围盒最小的（按钮文字应归按钮，而非外层容器）
    for ln in lines:
        la = {"x": ln["x"] * sx, "y": ln["y"] * sy, "w": ln["w"] * sx, "h": ln["h"] * sy}
        cands = [r for r in regions if _overlap_area(la, r) > 0.25 * la["w"] * la["h"]]
        if cands:
            best = min(cands, key=lambda r: r["w"] * r["h"])
            best["text"] = (best["text"] + " " + ln["text"]).strip()
        else:
            ln["region"] = None

    # 角色划分
    for r in regions:
        if r.get("shape", "").startswith("图标"):
            r["role"] = "图标"
        elif r.get("shape", "").startswith(("线段", "箭头")):
            r["role"] = "连线"
        elif r["area"] > 0.2 * total:
            r["role"] = "内容容器"
        elif r["p_w"] > 70 and r["py"] < 25 and r["p_h"] < 30:
            r["role"] = "顶部栏"
        elif r["p_w"] > 70 and r["py"] > 72 and r["p_h"] < 30:
            r["role"] = "底部栏"
        elif r["p_h"] > 45 and r["p_w"] < 40 and (r["px"] < 8 or r["px"] + r["p_w"] > 92):
            r["role"] = "侧边栏"   # 必须贴着左右边缘，内容区的竖列卡片不算
        else:
            r["role"] = "内容区"

    desc = []
    for role in ("顶部栏", "底部栏", "侧边栏", "内容容器"):
        rs = sorted((r for r in regions if r["role"] == role), key=lambda r: r["py"])
        for r in rs:
            label = f"{role}(y≈{r['py']:.0f}%)"
            if r["text"]:
                label += f"「{r['text']}」"
            desc.append(label)

    conn = [r for r in regions if r["role"] == "连线"]
    if conn:
        kinds = {}
        for r in conn:
            kinds[r["shape"]] = kinds.get(r["shape"], 0) + 1
        desc.append("连线: " + "、".join(f"{k}×{v}" for k, v in kinds.items()))

    icons = [r for r in regions if r["role"] == "图标"]
    if icons:
        kinds = {}
        for r in icons:
            name = r["shape"].split(":", 1)[1]
            kinds[name] = kinds.get(name, 0) + 1
        desc.append("图标: " + "、".join(f"{k}×{v}" for k, v in kinds.items()))

    body = [r for r in regions if r["role"] == "内容区"]
    rows = []
    for r in sorted(body, key=lambda r: r["py"]):
        for row in rows:
            if abs(r["py"] - row[0]["py"]) < 15:
                row.append(r)
                break
        else:
            rows.append([r])
    for i, row in enumerate(rows):
        names = []
        for r in sorted(row, key=lambda r: r["px"]):
            label = r["shape"]
            if r["text"]:
                label += f"「{r['text']}」"
            names.append(label)
        desc.append(f"第{i + 1}行(y≈{row[0]['py']:.0f}%): " + "、".join(names))
    return "\n".join(desc)


# ---------------- 彩色字符画 ----------------

def char_art(pil_rgb, width=72, plain=False, palette=None):
    """字符画：亮度决定字符，颜色用调色板量化后以色码标记（节省 token）"""
    tw = max(20, int(width))
    th = max(8, int(pil_rgb.height / pil_rgb.width * tw * 0.5))
    img = pil_rgb.resize((tw, th))
    px = img.load()
    rows = []
    for y in range(th):
        row, prev = [], None
        for x in range(tw):
            r, g, b = px[x, y]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ch = CHARS[min(9, int(lum / 25.6))]
            if palette:
                cur = min(palette, key=lambda c: abs(c[0] - r) + abs(c[1] - g) + abs(c[2] - b))
            else:
                cur = (r, g, b)
            if prev is None or cur != prev:
                if plain:
                    row.append(f"[{hex_of(cur)}]")
                else:
                    row.append(f"\x1b[38;2;{cur[0]};{cur[1]};{cur[2]}m")
                prev = cur
            row.append(ch)
        if not plain:
            row.append("\x1b[0m")
        rows.append("".join(row))
    return "\n".join(rows)


def compute_parents(regions):
    """元素层级：为每个区域找"最小的完全包含它的更大区域"作为父容器。
    卡片/容器内的按钮、图标、文字块由此建立归属关系"""
    for i, r in enumerate(regions):
        best = None
        for k in regions:
            if k is r or k["area"] <= r["area"] * 1.2:
                continue
            if (k["x"] <= r["x"] + 2 and k["y"] <= r["y"] + 2 and
                    k["x"] + k["w"] >= r["x"] + r["w"] - 2 and
                    k["y"] + k["h"] >= r["y"] + r["h"] - 2):
                if best is None or k["area"] < best["area"]:
                    best = k
        r["parent"] = best


def normalize_math(text):
    """数学符号规范化（高置信度替换）。返回 (规范化文本, 是否公式行)。
    仅当行含数学特征时替换，避免误伤普通文本"""
    import re
    if not re.search(r"[=<>≥≤×÷√αβγπ±]|f\s*\(|x\s*[²³⁰¹⁴⁵⁶⁷⁸⁹]|函数|不等式|方程|实根|恒成立|取值范围", text):
        return text, False
    t = text
    t = re.sub(r"(?<![A-Za-z0-9])x\s*2\b", "x²", t)
    t = re.sub(r"(?<![A-Za-z0-9])x\s*3\b", "x³", t)
    t = re.sub(r"(?<![A-Za-z0-9])x\s*4\b", "x⁴", t)
    t = t.replace("亠", "≥").replace("冫", "≥").replace("艹", "≥")
    t = t.replace("≥", "≥").replace("≤", "≤").replace("＞", ">").replace("＜", "<")
    t = re.sub(r"(?<=[\d\w)])\s*一\s*(?=[\d\w(])", "−", t)
    t = t.replace("（ l)", "（1)").replace("( l)", "(1)").replace("（ 1 ）", "（1）")
    return t, True


def guess_function(r):
    """基于形状/角色/文字猜测 UI 元素功能（明确标注为猜测，供模型参考）"""
    shape = r["shape"]
    text = (r.get("text") or "").strip()
    role = r.get("role", "")
    if shape.startswith("图标"):
        return {"搜索": "搜索入口", "设置": "设置", "菜单": "打开菜单", "关闭": "关闭/返回",
                "返回": "返回上级", "加号": "新建/添加"}.get(shape.split(":", 1)[1], "图标按钮")
    if shape.startswith("按钮"):
        for kw, fn in (("登录", "登录"), ("搜索", "搜索"), ("开始", "开始主操作"), ("立即", "立即执行"),
                       ("领取", "领取优惠"), ("播放", "播放"), ("去下单", "下单"), ("更多", "查看更多"),
                       ("换一换", "换一批"), ("编辑", "编辑"), ("保存", "保存"), ("发送", "发送"),
                       ("收藏", "收藏"), ("导航", "导航"), ("下载", "下载")):
            if kw in text:
                return f"按钮：{fn}"
        return "按钮（点击操作）"
    if shape.startswith("输入框"):
        return "输入/搜索框" + (f"（占位：{text[:12]}）" if text else "")
    if shape.startswith("卡片"):
        return "内容卡片" + (f"（{text[:16]}）" if text else "")
    if shape.startswith("徽章"):
        return "状态徽章/开关"
    if shape.startswith("指示点"):
        return "指示点/状态圆点"
    if shape.startswith("矩形框"):
        return "描边容器/输入框"
    if role == "顶部栏":
        return "页面标题栏"
    if role == "底部栏":
        return "底部导航栏"
    if role == "侧边栏":
        return "侧边导航"
    if role == "内容容器":
        return "内容承载区"
    if role == "连线":
        return "分隔线/连线" if "线段" in shape else "指向箭头"
    if shape.startswith("圆形"):
        return "圆形控件/状态点"
    if shape.startswith("菱形"):
        return "决策/标志形状"
    return "色块/内容区"


# ---------------- 主流程 ----------------

def cluster_text_blocks(lines, aW):
    """把文字行按垂直间距聚类成文本块（段落/卡片/弹窗内的文字组）"""
    if not lines:
        return []
    ls = sorted(lines, key=lambda l: (l["y"], l["x"]))
    heights = sorted(l["h"] for l in ls)
    med_h = heights[len(heights) // 2]
    gap_thr = max(12, med_h * 1.4)
    blocks, cur = [], [ls[0]]
    for l in ls[1:]:
        if l["y"] - (cur[-1]["y"] + cur[-1]["h"]) <= gap_thr:
            cur.append(l)
        else:
            blocks.append(cur)
            cur = [l]
    blocks.append(cur)
    out = []
    for blk in blocks:
        x1 = min(l["x"] for l in blk)
        y1 = min(l["y"] for l in blk)
        x2 = max(l["x"] + l["w"] for l in blk)
        y2 = max(l["y"] + l["h"] for l in blk)
        out.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                    "lines": [l["text"] for l in blk]})
    return out


def confidence_summary(lines, regions):
    """识别置信度：基于 OCR 行数与规则形状占比，防止模型拿噪声数据瞎编"""
    words_n = len(lines)
    irregular = sum(1 for r in regions if r["shape"] == "不规则形状")
    rule_n = len(regions) - irregular
    if words_n >= 5 and rule_n >= 3 and irregular <= len(regions) * 0.5:
        return (f"> 识别置信度：高 —— OCR 提取 {words_n} 行文字、{rule_n} 个规则区域为主，"
                f"数据可信，可直接引用。")
    if words_n >= 2 or rule_n >= 2:
        return (f"> 识别置信度：中 —— OCR {words_n} 行文字、{len(regions)} 个区域，"
                f"部分数据可用（注意引擎误读），形状/颜色有噪声。")
    return (f"> 识别置信度：低 —— OCR 无/极少文字（{words_n} 行），区域多为不规则色块，"
            f"疑似照片/小字/插画：仅配色与版面可信，内容需视觉模型确认。")


def _file_hash(path):
    """文件内容 MD5（用于缓存：同一张图无需重复分析）"""
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path_for(img_path, args):
    if getattr(args, "no_cache", False):
        return None
    try:
        cache_dir = Path(getattr(args, "cache_dir", None) or
                         os.path.join(os.path.expanduser("~"), ".cache", "image-analysis"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / (_file_hash(img_path) + ".md")
    except Exception:
        return None


def analyze_one(img_path, args, print_report=True):
    img_path = os.path.abspath(img_path)
    if not os.path.exists(img_path):
        print(f"错误: 文件不存在 {img_path}", file=sys.stderr)
        return 1

    # 缓存：相同内容的图片直接复用上次报告（同一张图再次提及时秒回，不重复分析）
    cache_path = _cache_path_for(img_path, args)
    if cache_path is not None and cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / (Path(img_path).stem + "_分析.md")).write_text(text, encoding="utf-8")
        if print_report:
            print(text)
        print(f"\n[缓存命中] {img_path}", file=sys.stderr)
        return 0

    with Image.open(img_path) as im:
        pil = ImageOps.exif_transpose(im).convert("RGB")
    ow, oh = pil.size

    # 1) OCR（多引擎并行 + 双遍放大）
    words = ocr_all(img_path, args.ocr)
    lines = merge_words_to_lines(words) if words else []
    ocr_note = ""
    if not words:
        ocr_note = "\n> ⚠️ OCR 未能提取文字（引擎不可用或图片无文字），以下为形状/颜色分析结果。"

    # 2) 形状/颜色分析（缩放图，保证坐标一致；直接用 RGB 数组，
    #    OpenCV 的轮廓/kmeans 等函数不关心通道顺序，避免 BGR/RGB 颜色错位）
    s = min(1.0, MAX_ANALYSIS_DIM / max(ow, oh))
    aW, aH = max(1, int(ow * s)), max(1, int(oh * s))
    img_rgb = np.array(pil.resize((aW, aH), Image.LANCZOS))
    regions, palette, bg_regions = detect_regions(img_rgb, aW, aH)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # 文字重叠过滤用单词级框（行级框会把同一行多个词连成一大片，
    # 误伤从文字行下方穿过的真实线条）
    regions += detect_lines(gray, regions,
                            [(w["x"] * s - 12, w["y"] * s - 12,
                              w["w"] * s + 24, w["h"] * s + 24) for w in words],
                            aW, aH)
    regions += detect_icons(gray, aW, aH,
                            [(w["x"] * s, w["y"] * s, w["w"] * s, w["h"] * s, w["text"]) for w in words],
                            regions)
    for r in regions:
        r.setdefault("color_hex", hex_of(r["color"]))
        r.setdefault("color_name", color_name(r["color"]))
    regions.sort(key=lambda r: (r["y"], r["x"]))
    classify_controls(regions, aW, aH)   # 控件语义标注（在布局分析前，pct 自行计算）

    # 3) 布局与文字挂接
    layout_desc = analyze_layout(regions, lines, aW, aH, ow, oh)
    compute_parents(regions)   # 元素层级：父容器归属

    # 3.5) 表格网格识别（OCR 单词转分析图坐标）
    tables = []
    if words:
        words_an = [{"text": w["text"], "x": w["x"] * s, "y": w["y"] * s,
                     "w": w["w"] * s, "h": w["h"] * s} for w in words]
        tables = detect_tables(gray, words_an, aW, aH)

    # 3.6) 文本块聚类
    blocks = cluster_text_blocks(lines, aW)

    # 4) 字符画（用分析调色板量化颜色，减少色码数量）
    art = "" if args.no_ascii else char_art(pil, args.width, args.plain, palette)

    if args.debug:
        for r in regions:
            print(f"[debug] {r['role']} {r['shape']} ({r['x']},{r['y']}) {r['w']}x{r['h']} "
                  f"{r['color_hex']} fill={r['fill']:.2f} text={r['text']!r}", file=sys.stderr)

    # 5) 组装 markdown（七段式，编号自动递增）
    cn = ["一", "二", "三", "四", "五", "六", "七", "八"]
    sections = []

    # 一、整体信息
    body = [f"- 原图尺寸: {ow}×{oh} px"]
    seen = {}
    for r in regions:
        if r.get("role") in ("连线", "图标"):   # 线/图标是描边不是色块，不计面积
            continue
        seen[r["color_hex"]] = seen.get(r["color_hex"], 0) + r["area"]
    bg_seen = {}
    for r in bg_regions:                        # 背景色也计入主色（标注背景）
        bg_seen[r["color_hex"]] = bg_seen.get(r["color_hex"], 0) + r["area"]
    if seen or bg_seen:
        body.append("- 主要颜色（按面积）:")
        items = [(k, v, False) for k, v in seen.items()] \
            + [(k, v, True) for k, v in bg_seen.items()]
        for k, v, is_bg in sorted(items, key=lambda kv: -kv[1])[:6]:
            body.append(f"  - {k} {color_name(rgb_of_hex(k))}{'（背景）' if is_bg else ''} — "
                        f"{v / (aW * aH) * 100:.0f}%")
    else:
        body.append("- 未识别出色块区域")
    sections.append(("整体信息", body))

    # 二、元素清单
    id_map = {id(r): i for i, r in enumerate(regions, 1)}
    body = ["| # | 所属 | 功能(猜测) | 类型/形状 | x范围 | y范围 | 大小 | 颜色 | 包含文字 |",
            "|---|------|-----------|----------|-------|-------|------|------|---------|"]
    if regions:
        table_rows = []
        for i, r in enumerate(regions, 1):
            text = (r["text"] or "—").replace("|", "\\|")
            parent = f"#{id_map[id(r['parent'])]}" if r["parent"] else "—"
            table_rows.append(f"| {i} | {parent} | {guess_function(r)} | {r['shape']} | "
                              f"{r['px']:.0f}-{r['px'] + r['p_w']:.0f}% | "
                              f"{r['py']:.0f}-{r['py'] + r['p_h']:.0f}% | "
                              f"{r['p_w']:.0f}×{r['p_h']:.0f}% | "
                              f"{r['color_hex']} {r['color_name']} | {text} |")
        if len(table_rows) > 15:
            body.extend(table_rows[:15])
            body.append(f"| … | | 其余 {len(table_rows) - 15} 个区域省略 | | | | | | |")
        else:
            body.extend(table_rows)
    else:
        body.append("| — | — | 未识别出明显色块 | — | — | — | — | — | — |")
    sections.append(("识别到的元素（坐标为百分比 %）", body))

    # 三、布局结构
    sections.append(("布局结构", [layout_desc if layout_desc else "- 无明显框架结构"]))

    # 三.五、层级关系（父子包含树）
    children = {}
    for r in regions:
        if r["parent"]:
            children.setdefault(id(r["parent"]), []).append(r)
    if children:
        tree = []
        for r in regions:
            kids = children.get(id(r))
            if kids:
                tree.append(f"- {r['shape']}#{id_map[id(r)]} "
                            f"（x {r['px']:.0f}-{r['px'] + r['p_w']:.0f}%, "
                            f"y {r['py']:.0f}-{r['py'] + r['p_h']:.0f}%）包含: "
                            + "、".join(f"{k['shape']}#{id_map[id(k)]}" for k in kids))
        sections.append(("层级关系", tree))

    # 四、文本块聚类
    if blocks:
        body = []
        for i, b in enumerate(blocks, 1):
            body.append(f"### 文本块 {i}（x {b['x'] / ow * 100:.0f}-{(b['x'] + b['w']) / ow * 100:.0f}%, "
                        f"y {b['y'] / oh * 100:.0f}-{(b['y'] + b['h']) / oh * 100:.0f}%）")
            for j, t in enumerate(b["lines"], 1):
                body.append(f"{j}. \"{t}\"")
        sections.append(("文本块聚类", body))

    # 五、表格识别
    if tables:
        body = []
        for i, t in enumerate(tables, 1):
            body += [f"### 表格 {i}", t]
        sections.append(("表格识别", body))

    # 六、彩色字符画
    if not args.no_ascii:
        body = ["（" + ("每字符一格，`[#RRGGBB]` 为颜色标签" if args.plain
                        else "每字符一格，ANSI 色码标记颜色，如 `\\x1b[38;2;R;G;Bm`") + "）",
                "```", art, "```"]
        sections.append(("彩色字符画", body))

    # 七、OCR 全文（数学行做符号规范化并标注）
    body = []
    if lines:
        for ln in lines:
            t, is_math = normalize_math(ln["text"])
            tag = "  [公式，符号可能误读，需数学语境理解]" if is_math else ""
            body.append(f"- \"{t}\"{tag} (x={ln['x'] / ow * 100:.0f}%, y={ln['y'] / oh * 100:.0f}%)")
    else:
        body.append("- （无）")
    if ocr_note:
        body.append(ocr_note)
    sections.append(("完整文字内容（OCR）", body))

    parts = ["# 图片分析结果", confidence_summary(lines, regions), ""]
    for i, (title, body) in enumerate(sections):
        parts.append(f"## {cn[i]}、{title}")
        parts.extend(body)
        parts.append("")

    text = "\n".join(parts)
    if print_report:
        print(text)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / (Path(img_path).stem + "_分析.md")
    else:
        out = Path(args.out or (Path(img_path).stem + "_分析.md"))
    out.write_text(text, encoding="utf-8")
    if cache_path is not None:
        try:
            cache_path.write_text(text, encoding="utf-8")
        except Exception:
            pass
    if print_report:
        print(text)
        print(f"\n[已保存] {os.path.abspath(out)}", file=sys.stderr)
    return 0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="图片结构化分析：OCR + 形状/颜色 + 布局 + 文本块 + 表格 + 彩色字符画")
    ap.add_argument("images", nargs="+", help="图片路径（可多个），或目录（批量分析其中全部图片）")
    ap.add_argument("--width", type=int, default=72, help="字符画宽度（默认 72）")
    ap.add_argument("--plain", action="store_true", help="字符画用 [十六进制色码] 代替 ANSI 色码")
    ap.add_argument("--no-ascii", action="store_true", help="不生成字符画")
    ap.add_argument("--out", help="单图输出文件（默认: 图片名_分析.md）")
    ap.add_argument("--out-dir", help="批量输出目录（多图时建议指定）")
    ap.add_argument("--ocr", choices=["auto", "windows", "rapid", "both"], default="auto",
                    help="OCR 引擎：auto=Windows+(已装则加 RapidOCR)并行；windows/rapid=单引擎；both=强制双引擎")
    ap.add_argument("--debug", action="store_true", help="输出调试信息")
    ap.add_argument("--workers", type=int, default=3, help="并行分析的工作进程数（多图时生效，默认 3）")
    ap.add_argument("--cache-dir", help="缓存目录（默认 ~/.cache/image-analysis）")
    ap.add_argument("--no-cache", action="store_true", help="禁用缓存（同图重复分析）")
    args = ap.parse_args()

    targets = []
    for t in args.images:
        p = Path(t)
        if p.is_dir():
            targets += sorted(str(f) for f in p.iterdir()
                              if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
        else:
            targets.append(t)
    if not targets:
        sys.exit("未找到图片文件")

    # 多图并行（多进程）；单图直接分析
    failed = 0
    workers = max(1, min(args.workers, len(targets)))
    if len(targets) > 1 and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(analyze_one, t, args, False): t for t in targets}
            done = 0
            for fut in as_completed(futs):
                done += 1
                t = futs[fut]
                try:
                    failed += fut.result()
                except Exception as e:
                    print(f"错误: {t}: {e}", file=sys.stderr)
                    failed += 1
                print(f"[{done}/{len(targets)}] 完成 {t}", file=sys.stderr)
    else:
        for i, t in enumerate(targets, 1):
            if len(targets) > 1:
                print(f"\n[{i}/{len(targets)}] {t}", file=sys.stderr)
            try:
                failed += analyze_one(t, args, True)
            except Exception as e:
                print(f"错误: {t}: {e}", file=sys.stderr)
                failed += 1
    if failed:
        sys.exit(f"{failed} 个文件分析失败")


if __name__ == "__main__":
    main()
