#!/usr/bin/env python3
"""
Leave-for-work Kinect + OpenRouter checker.

Flow:
1) Watch depth frames from Kinect v1 for sustained close-range presence.
2) Fetch checklist JSON from CHECKLIST_URL:
     {"items": ["keys", "badge", "wallet"]}
3) Capture RGB frame and send it to OpenRouter vision model.
4) Expect JSON response:
     {"all_good": true|false, "missing": ["keys"]}
5) If missing items exist, play alert sound and POST webhook event.

Required environment variables:
- CHECKLIST_URL
- OPENROUTER_API_KEY

Recommended environment variables:
- OPENROUTER_MODEL (default: openai/gpt-4.1-mini)
- WEBHOOK_URL

Optional auth environment variables:
- CHECKLIST_BEARER_TOKEN
- CHECKLIST_AUTH_HEADER            (format: "Header-Name: value")
- WEBHOOK_BEARER_TOKEN
- WEBHOOK_API_KEY_HEADER           (header name, e.g. "x-api-key")
- WEBHOOK_API_KEY_VALUE

Optional audio environment variables:
- ALERT_SOUND_PATH                 (path to wav/mp3 file)
- ALERT_PLAYER_COMMAND             (format command; use {path} placeholder)
"""

import argparse
import base64
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import freenect
import httpx
import numpy as np


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def env_header(header_line: str) -> Dict[str, str]:
    if ":" not in header_line:
        raise ValueError("Expected HEADER format 'Name: value'")
    name, value = header_line.split(":", 1)
    name = name.strip()
    value = value.strip()
    if not name or not value:
        raise ValueError("Invalid header content")
    return {name: value}


def build_auth_headers(prefix: str) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    bearer = os.getenv(f"{prefix}_BEARER_TOKEN")
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    raw = os.getenv(f"{prefix}_AUTH_HEADER")
    if raw:
        headers.update(env_header(raw))
    return headers


def build_webhook_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    bearer = os.getenv("WEBHOOK_BEARER_TOKEN")
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"

    key_name = os.getenv("WEBHOOK_API_KEY_HEADER")
    key_value = os.getenv("WEBHOOK_API_KEY_VALUE")
    if key_name and key_value:
        headers[key_name] = key_value
    return headers


def get_depth_frame() -> Optional[np.ndarray]:
    frame, _ = freenect.sync_get_depth()
    if frame is None:
        return None
    return frame.astype(np.uint16)


def center_roi(depth: np.ndarray, roi_frac: float) -> np.ndarray:
    h, w = depth.shape
    fh = max(1, int(h * roi_frac))
    fw = max(1, int(w * roi_frac))
    y1 = (h - fh) // 2
    x1 = (w - fw) // 2
    return depth[y1 : y1 + fh, x1 : x1 + fw]


def is_close(depth: np.ndarray, roi_frac: float, close_threshold_mm: int) -> Tuple[bool, int]:
    roi = center_roi(depth, roi_frac)
    valid = roi[(roi > 0) & (roi < 2047)]
    if valid.size == 0:
        return False, 9999
    p10 = int(np.percentile(valid, 10))
    return p10 <= close_threshold_mm, p10


def capture_rgb_data_url(jpeg_quality: int) -> Optional[str]:
    frame, _ = freenect.sync_get_video()
    if frame is None:
        return None
    rgb = np.asarray(frame)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        return None
    encoded = base64.b64encode(jpg.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def save_debug_frame(data_url: str, out_dir: str) -> Optional[str]:
    if not data_url.startswith("data:image/jpeg;base64,"):
        return None
    payload = data_url.split(",", 1)[1]
    raw = base64.b64decode(payload)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"leave_check_{ts}.jpg"
    path.write_bytes(raw)
    return str(path)


def fetch_checklist_items(client: httpx.Client, checklist_url: str) -> List[str]:
    headers = build_auth_headers("CHECKLIST")
    response = client.get(checklist_url, headers=headers)
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Checklist response missing list field: items")
    normalized = [str(item).strip() for item in items if str(item).strip()]
    if not normalized:
        raise ValueError("Checklist items list is empty")
    return normalized


def extract_json_blob(raw_text: str) -> Optional[Dict[str, Any]]:
    try:
        loaded = json.loads(raw_text)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        loaded = json.loads(raw_text[start : end + 1])
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        return None
    return None


def call_openrouter_vision(
    client: httpx.Client, api_key: str, model: str, checklist_items: List[str], image_data_url: str
) -> Dict[str, Any]:
    checklist_text = "\n".join(f"- {item}" for item in checklist_items)
    prompt_text = (
        "Check this photo and determine if the person appears to have the required items.\n"
        "Return STRICT JSON only with keys: all_good (boolean) and missing (array of strings).\n"
        "Checklist:\n"
        f"{checklist_text}"
    )

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You validate leave-for-work checklist items from a single photo. "
                    "Always answer in compact JSON only."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = client.post(OPENROUTER_URL, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    message = data["choices"][0]["message"]["content"]
    if isinstance(message, list):
        message = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in message
        )
    if not isinstance(message, str):
        raise ValueError("Unexpected model response type")

    parsed = extract_json_blob(message)
    if not parsed:
        raise ValueError("Model response did not contain valid JSON object")
    return parsed


def parse_missing_items(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
    all_good = bool(result.get("all_good", False))
    missing_raw = result.get("missing", [])
    if not isinstance(missing_raw, list):
        missing_raw = []
    missing = [str(item).strip() for item in missing_raw if str(item).strip()]
    if all_good and not missing:
        return True, []
    if not all_good and missing:
        return False, missing
    if missing:
        return False, missing
    return True, []


def play_alert_sound() -> None:
    sound_path = os.getenv("ALERT_SOUND_PATH")
    custom = os.getenv("ALERT_PLAYER_COMMAND")

    if custom and sound_path:
        command = [part.format(path=sound_path) for part in shlex.split(custom)]
        subprocess.Popen(command)  # nosec B603
        return

    if sound_path:
        players = [
            ["aplay", sound_path],
            ["paplay", sound_path],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", sound_path],
        ]
        for cmd in players:
            if shutil.which(cmd[0]):
                subprocess.Popen(cmd)  # nosec B603
                return

    # Final fallback, audible bell if terminal supports it.
    print("\a", end="", flush=True)


def post_webhook_event(
    client: httpx.Client,
    webhook_url: str,
    model: str,
    checklist_url: str,
    missing_items: List[str],
) -> None:
    checklist_host = urlparse(checklist_url).netloc or checklist_url
    payload = {
        "event": "leave_check_forgot_items",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "missing": missing_items,
        "checklist_source": checklist_host,
        "model": model,
    }
    headers = build_webhook_headers()
    response = client.post(webhook_url, json=payload, headers=headers)
    response.raise_for_status()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect close-range leave events with Kinect depth, verify required items with "
            "OpenRouter vision, then play an alert and notify webhook when items are missing."
        )
    )
    parser.add_argument("--roi-frac", type=float, default=0.45, help="Center ROI fraction (0.1-1.0)")
    parser.add_argument(
        "--close-threshold-mm",
        type=int,
        default=1100,
        help="Close threshold on p10 depth value from center ROI",
    )
    parser.add_argument(
        "--min-close-frames",
        type=int,
        default=8,
        help="Consecutive close frames required before triggering check",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=120,
        help="Seconds between full leave checks",
    )
    parser.add_argument(
        "--http-timeout-seconds",
        type=float,
        default=20.0,
        help="HTTP timeout for checklist/OpenRouter/webhook",
    )
    parser.add_argument("--jpeg-quality", type=int, default=88, help="RGB JPEG quality (1-100)")
    parser.add_argument(
        "--save-debug-dir",
        default="",
        help="If provided, save each trigger image as JPEG in this directory",
    )
    parser.add_argument("--debug-depth", action="store_true", help="Print depth stats during loop")
    return parser.parse_args()


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def run_loop(args: argparse.Namespace) -> None:
    checklist_url = required_env("CHECKLIST_URL")
    api_key = required_env("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini").strip()
    webhook_url = os.getenv("WEBHOOK_URL", "").strip()

    if not 0.1 <= args.roi_frac <= 1.0:
        raise RuntimeError("--roi-frac must be between 0.1 and 1.0")
    if not 1 <= args.jpeg_quality <= 100:
        raise RuntimeError("--jpeg-quality must be between 1 and 100")

    close_frames = 0
    last_trigger = 0.0
    timeout = httpx.Timeout(args.http_timeout_seconds)

    print("Leave-check loop started.")
    print(f"Checklist URL: {checklist_url}")
    print(f"OpenRouter model: {model}")
    if webhook_url:
        print(f"Webhook enabled: {webhook_url}")
    else:
        print("Webhook disabled (WEBHOOK_URL not set).")
    print("Press Ctrl+C to stop.")

    with httpx.Client(timeout=timeout) as client:
        while True:
            depth = get_depth_frame()
            if depth is None:
                time.sleep(0.05)
                continue

            close_now, p10 = is_close(
                depth=depth,
                roi_frac=args.roi_frac,
                close_threshold_mm=args.close_threshold_mm,
            )
            close_frames = close_frames + 1 if close_now else 0

            if args.debug_depth:
                print(f"[depth] p10={p10} close={close_now} frames={close_frames}")

            now = time.time()
            if close_frames < args.min_close_frames:
                time.sleep(0.02)
                continue

            if now - last_trigger < args.cooldown_seconds:
                if args.debug_depth:
                    left = int(args.cooldown_seconds - (now - last_trigger))
                    print(f"[cooldown] {left}s remaining")
                time.sleep(0.1)
                continue

            last_trigger = now
            close_frames = 0
            print("[trigger] Running leave-check pipeline...")

            try:
                checklist_items = fetch_checklist_items(client, checklist_url)
            except Exception as exc:
                print(f"[error] checklist fetch failed: {exc}")
                time.sleep(0.2)
                continue

            image_data_url = capture_rgb_data_url(args.jpeg_quality)
            if not image_data_url:
                print("[error] failed to capture RGB frame")
                time.sleep(0.2)
                continue

            if args.save_debug_dir:
                debug_path = save_debug_frame(image_data_url, args.save_debug_dir)
                if debug_path:
                    print(f"[debug] saved frame: {debug_path}")

            try:
                result = call_openrouter_vision(
                    client=client,
                    api_key=api_key,
                    model=model,
                    checklist_items=checklist_items,
                    image_data_url=image_data_url,
                )
            except Exception as exc:
                print(f"[error] OpenRouter check failed: {exc}")
                time.sleep(0.2)
                continue

            all_good, missing = parse_missing_items(result)
            if all_good:
                print("[ok] No missing items detected.")
                time.sleep(0.2)
                continue

            print(f"[alert] Missing items: {missing}")
            play_alert_sound()

            if webhook_url:
                try:
                    post_webhook_event(
                        client=client,
                        webhook_url=webhook_url,
                        model=model,
                        checklist_url=checklist_url,
                        missing_items=missing,
                    )
                    print("[webhook] Event sent.")
                except Exception as exc:
                    print(f"[error] webhook send failed: {exc}")

            time.sleep(0.2)


def main() -> int:
    args = parse_args()
    try:
        run_loop(args)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    except Exception as exc:
        print(f"Fatal: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
