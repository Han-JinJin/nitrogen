# -*- coding: utf-8 -*-
"""
YouTube 批量下载 + 每20秒分段脚本
使用方法:
    python yt_batch_download.py                          # 交互模式，粘贴URL
    python yt_batch_download.py urls.txt                 # 从文件读取URL列表
    python yt_batch_download.py "https://..." "https://..."  # 直接传URL
"""

import subprocess
import os
import sys
import re
import json
import math
import platform
import shutil
from pathlib import Path

# ============== 配置区 ==============
# 脚本所在目录作为基准路径，方便部署
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IS_WINDOWS = platform.system() == "Windows"

# yt-dlp: 优先用同目录下的，否则找系统 PATH
_ytdlp_local = os.path.join(BASE_DIR, "yt-dlp.exe" if IS_WINDOWS else "yt-dlp")
YT_DLP = _ytdlp_local if os.path.isfile(_ytdlp_local) else shutil.which("yt-dlp") or "yt-dlp"

# cookies: 同目录下的 cookies.txt
COOKIES = os.path.join(BASE_DIR, "cookies.txt")

# ffmpeg 目录: 同目录下的 ffmpeg/ 或系统 PATH
FFMPEG_DIR = os.path.join(BASE_DIR, "ffmpeg")

OUTPUT_DIR = os.path.join(BASE_DIR, "downloads")   # 下载总目录
SEGMENT_DURATION = 20                               # 每段秒数
PROXY = "http://127.0.0.1:7890"                     # 代理地址
# ====================================


def get_ffmpeg_path():
    """获取 ffmpeg 路径，优先同目录，其次系统 PATH，最后自动下载"""
    ffmpeg_name = "ffmpeg.exe" if IS_WINDOWS else "ffmpeg"
    ffprobe_name = "ffprobe.exe" if IS_WINDOWS else "ffprobe"

    # 1) 检查 FFMPEG_DIR
    if os.path.isfile(os.path.join(FFMPEG_DIR, ffmpeg_name)):
        return FFMPEG_DIR

    # 2) 检查系统 PATH
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        print(f"[+] 使用系统 ffmpeg: {sys_ffmpeg}")
        return os.path.dirname(sys_ffmpeg)

    # 3) 自动下载
    print("[*] ffmpeg 未找到，正在自动下载...")
    os.makedirs(FFMPEG_DIR, exist_ok=True)

    try:
        import urllib.request

        if IS_WINDOWS:
            import zipfile
            url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            archive = os.path.join(FFMPEG_DIR, "ffmpeg.zip")
            print(f"[*] 下载中(~80MB)... {url}")
            urllib.request.urlretrieve(url, archive)
            with zipfile.ZipFile(archive, 'r') as zf:
                for member in zf.namelist():
                    bn = os.path.basename(member)
                    if bn in ("ffmpeg.exe", "ffprobe.exe"):
                        with zf.open(member) as src, open(os.path.join(FFMPEG_DIR, bn), 'wb') as dst:
                            dst.write(src.read())
            os.remove(archive)
        else:
            # Linux: 下载静态编译版本
            import tarfile
            url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
            archive = os.path.join(FFMPEG_DIR, "ffmpeg.tar.xz")
            print(f"[*] 下载中(~70MB)... {url}")
            urllib.request.urlretrieve(url, archive)
            with tarfile.open(archive, "r:xz") as tf:
                for member in tf.getmembers():
                    bn = os.path.basename(member.name)
                    if bn in ("ffmpeg", "ffprobe"):
                        member.name = bn
                        tf.extract(member, FFMPEG_DIR)
                        os.chmod(os.path.join(FFMPEG_DIR, bn), 0o755)
            os.remove(archive)

        if os.path.isfile(os.path.join(FFMPEG_DIR, ffmpeg_name)):
            print("[+] ffmpeg 下载完成!")
            return FFMPEG_DIR
    except Exception as e:
        print(f"[!] 自动下载失败: {e}")

    print("=" * 60)
    print("[!] 请手动安装 ffmpeg:")
    if IS_WINDOWS:
        print("    访问 https://www.gyan.dev/ffmpeg/builds/")
        print(f"    将 ffmpeg.exe, ffprobe.exe 放到 {FFMPEG_DIR}")
    else:
        print("    Ubuntu/Debian: sudo apt install ffmpeg")
        print("    CentOS/RHEL:   sudo yum install ffmpeg")
        print(f"    或下载静态版放到 {FFMPEG_DIR}")
    print("=" * 60)
    sys.exit(1)


def sanitize_filename(name):
    """清理文件名中的非法字符"""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name[:150]  # 限制长度


def get_video_info(url, ffmpeg_dir):
    """获取视频信息（标题、时长）"""
    cmd = [
        YT_DLP,
        "--cookies", COOKIES,
        "--ffmpeg-location", ffmpeg_dir,
        "--proxy", PROXY,
        "--dump-json",
        "--no-download",
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        print(f"[!] 获取视频信息失败: {result.stderr[:200]}")
        return None
    info = json.loads(result.stdout)
    return {
        "title": info.get("title", "unknown"),
        "duration": info.get("duration", 0),
        "id": info.get("id", "unknown"),
    }


def download_video(url, ffmpeg_dir):
    """下载完整视频"""
    info = get_video_info(url, ffmpeg_dir)
    if not info:
        print(f"[!] 跳过: {url}")
        return None

    title = sanitize_filename(info["title"])
    video_id = info["id"]
    duration = info["duration"]

    print(f"\n{'='*60}")
    print(f"[>] 标题: {title}")
    print(f"[>] 时长: {duration}s ({duration//60}分{duration%60}秒)")
    print(f"[>] ID:   {video_id}")
    print(f"{'='*60}")

    # 为每个视频创建独立目录
    video_dir = os.path.join(OUTPUT_DIR, f"{title}_{video_id}")
    full_dir = os.path.join(video_dir, "full")        # 完整视频
    segments_dir = os.path.join(video_dir, "segments") # 分段视频
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(segments_dir, exist_ok=True)

    # 下载完整视频
    output_template = os.path.join(full_dir, f"{title}.%(ext)s")
    cmd = [
        YT_DLP,
        "--cookies", COOKIES,
        "--ffmpeg-location", ffmpeg_dir,
        "--proxy", PROXY,
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--progress",
        url
    ]

    print(f"[*] 正在下载完整视频...")
    result = subprocess.run(cmd, encoding="utf-8")
    if result.returncode != 0:
        print(f"[!] 下载失败: {url}")
        return None

    # 找到下载的文件
    video_file = None
    for f in os.listdir(full_dir):
        if f.endswith(".mp4"):
            video_file = os.path.join(full_dir, f)
            break

    if not video_file:
        print("[!] 未找到下载的视频文件")
        return None

    print(f"[+] 下载完成: {video_file}")

    # 切割成 20 秒片段
    split_video(video_file, segments_dir, duration, ffmpeg_dir)

    return video_dir


def split_video(video_file, segments_dir, duration, ffmpeg_dir):
    """将视频切割为每 SEGMENT_DURATION 秒一段"""
    ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe" if IS_WINDOWS else "ffmpeg")

    if duration <= 0:
        # 如果没拿到时长，用 ffprobe 获取
        ffprobe_exe = os.path.join(ffmpeg_dir, "ffprobe.exe" if IS_WINDOWS else "ffprobe")
        cmd = [
            ffprobe_exe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            duration = float(result.stdout.strip())
        except ValueError:
            print("[!] 无法获取视频时长，跳过切割")
            return

    total_segments = math.ceil(duration / SEGMENT_DURATION)
    basename = Path(video_file).stem

    print(f"[*] 正在切割为 {total_segments} 个片段 (每段 {SEGMENT_DURATION}s)...")

    for i in range(total_segments):
        start = i * SEGMENT_DURATION
        segment_file = os.path.join(
            segments_dir,
            f"{basename}_seg{i+1:04d}_{start}s-{min(start+SEGMENT_DURATION, int(duration))}s.mp4"
        )

        cmd = [
            ffmpeg_exe,
            "-y",                       # 覆盖已有文件
            "-i", video_file,
            "-ss", str(start),          # 起始时间
            "-t", str(SEGMENT_DURATION),# 持续时间
            "-c", "copy",               # 直接复制，不重新编码（极快）
            "-avoid_negative_ts", "make_zero",
            segment_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"    [+] 片段 {i+1}/{total_segments}: {os.path.basename(segment_file)}")
        else:
            print(f"    [!] 片段 {i+1} 切割失败: {result.stderr[:100]}")

    print(f"[+] 切割完成! 共 {total_segments} 个片段 -> {segments_dir}")


def read_urls_from_file(filepath):
    """从文件读取URL列表"""
    urls = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "youtube.com" in line or "youtu.be" in line:
                urls.append(line)
    return urls


def interactive_input():
    """交互式输入URL"""
    print("请粘贴 YouTube 视频链接（每行一个，输入空行结束）:")
    urls = []
    while True:
        try:
            line = input().strip()
        except EOFError:
            break
        if not line:
            break
        if "youtube.com" in line or "youtu.be" in line:
            urls.append(line)
        else:
            print(f"  [!] 跳过非YouTube链接: {line[:50]}")
    return urls


def main():
    print("=" * 60)
    print("  YouTube 批量下载 + 20秒分段工具")
    print("=" * 60)

    # 确保 ffmpeg 可用
    ffmpeg_dir = get_ffmpeg_path()
    print(f"[+] ffmpeg: {ffmpeg_dir}")
    print(f"[+] yt-dlp: {YT_DLP}")
    print(f"[+] cookies: {COOKIES}")
    print(f"[+] 输出目录: {OUTPUT_DIR}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 收集URL
    urls = []
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if os.path.isfile(arg):
                # 参数是文件路径
                urls.extend(read_urls_from_file(arg))
                print(f"[*] 从文件 {arg} 读取了 {len(urls)} 个链接")
            else:
                # 参数是URL
                urls.append(arg)
    else:
        urls = interactive_input()

    if not urls:
        print("[!] 没有有效的URL，退出")
        return

    print(f"\n[*] 共 {len(urls)} 个视频待下载\n")

    # 逐个下载（避免被限速/封号）
    success = 0
    failed = 0
    for idx, url in enumerate(urls, 1):
        print(f"\n[{idx}/{len(urls)}] {url}")
        try:
            result = download_video(url, ffmpeg_dir)
            if result:
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[!] 异常: {e}")
            failed += 1

    # 汇总
    print(f"\n{'='*60}")
    print(f"  完成! 成功: {success}, 失败: {failed}")
    print(f"  文件保存在: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
