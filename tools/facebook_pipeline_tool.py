#!/usr/bin/env python3
"""Facebook content review pipeline tools.

This module supports a human-in-the-loop Facebook posting workflow:
- discover local user-provided media without reading image/video pixels,
- store review state durably,
- render Telegram inline-button review prompts,
- publish only after explicit approval.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import re
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

DEFAULT_MEDIA_DIR = get_hermes_home() / "automations" / "FacebookMedia"
STATE_DIR = get_hermes_home() / "automations" / "facebook_posts" / "state"
SOURCE_CONFIG = get_hermes_home() / "automations" / "facebook_daily_source.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi"}
SUPPORTED_EXTS = IMAGE_EXTS | VIDEO_EXTS
GENERIC_MEDIA_MATCH_TOKENS = {
    "anh",
    "ảnh",
    "cay",
    "cây",
    "chau",
    "chậu",
    "dep",
    "đẹp",
    "la",
    "lá",
    "noi",
    "nội",
    "phong",
    "phòng",
    "that",
    "thất",
    "van",
    "văn",
    "xanh",
}

Publisher = Callable[..., Awaitable[Dict[str, Any]]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug_tokens(text: str) -> set[str]:
    raw = (text or "").lower()
    ascii_text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    tokens = set()
    for candidate in (raw, ascii_text):
        tokens.update(
            token
            for token in re.split(r"[^0-9a-zA-ZÀ-ỹ]+", candidate)
            if len(token) >= 2 and token not in GENERIC_MEDIA_MATCH_TOKENS
        )
    return tokens


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _state_path(review_id: str) -> Path:
    safe_id = re.sub(r"[^0-9A-Za-z_.-]", "_", review_id)
    return STATE_DIR / f"{safe_id}.json"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}: {exc}") from exc


def _normalize_media_metadata(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        if isinstance(raw.get("media"), list):
            items = raw["media"]
        elif isinstance(raw.get("files"), list):
            items = raw["files"]
        else:
            items = []
            for key, value in raw.items():
                if isinstance(value, dict):
                    entry = dict(value)
                    entry.setdefault("file", key)
                    items.append(entry)
    else:
        items = []
    return [item for item in items if isinstance(item, dict)]


def _metadata_by_filename(media_dir: Path) -> Dict[str, Dict[str, Any]]:
    metadata_path = media_dir / "media.json"
    raw = None
    if metadata_path.exists():
        try:
            raw = _read_json(metadata_path, None)
        except PermissionError:
            logger.warning("No permission to read media metadata: %s", metadata_path)
            raw = None
    result: Dict[str, Dict[str, Any]] = {}
    for item in _normalize_media_metadata(raw):
        filename = (
            item.get("file")
            or item.get("filename")
            or item.get("name")
            or item.get("path")
        )
        if not filename:
            continue
        result[Path(str(filename)).name] = item
    return result


def discover_media(media_dir: str | Path = DEFAULT_MEDIA_DIR) -> List[Dict[str, Any]]:
    """Return local media metadata without opening/reading media contents."""
    root = Path(media_dir).expanduser()
    if not root.exists():
        return []

    metadata = _metadata_by_filename(root)
    discovered: List[Dict[str, Any]] = []
    try:
        paths = sorted(root.rglob("*"))
    except PermissionError:
        logger.warning("No permission to scan media directory: %s", root)
        return []
    for path in paths:
        if not path.is_file() or path.name == "media.json":
            continue
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue
        stat = path.stat()
        meta = metadata.get(path.name, {})
        tags = meta.get("tags") or meta.get("keywords") or []
        if isinstance(tags, str):
            tags = [part.strip() for part in re.split(r"[,;]", tags) if part.strip()]
        item = {
            "filename": path.name,
            "path": str(path),
            "relative_path": str(path.relative_to(root)),
            "type": "video" if ext in VIDEO_EXTS else "image",
            "extension": ext,
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "description": meta.get("description") or meta.get("caption") or meta.get("note") or "",
            "tags": list(tags) if isinstance(tags, list) else [],
        }
        discovered.append(item)
    return discovered


def select_media(media_items: Iterable[Dict[str, Any]], hint: str = "") -> Optional[Dict[str, Any]]:
    """Select media only when explicit metadata tags match the content hint.

    Tags are treated as the user-provided source of truth. If no tag overlaps
    the source/post topic after dropping generic words, return None instead of
    falling back to newest media; an unrelated image is worse than no image.
    """
    items = list(media_items)
    if not items:
        return None
    hint_tokens = _slug_tokens(hint)

    def score(item: Dict[str, Any]) -> tuple[int, float]:
        tags = item.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        media_tokens = _slug_tokens(" ".join(str(t) for t in tags if t))
        overlap = len(hint_tokens & media_tokens) if hint_tokens else 0
        return (overlap, float(item.get("mtime") or 0.0))

    selected = max(items, key=score)
    best_score, _ = score(selected)
    if best_score < 1:
        return None
    return selected


def build_review_keyboard(review_id: str) -> Dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "✅ Duyệt 1", "callback_data": f"fbp:a:{review_id}:1"},
                {"text": "✅ Duyệt 2", "callback_data": f"fbp:a:{review_id}:2"},
                {"text": "✅ Duyệt 3", "callback_data": f"fbp:a:{review_id}:3"},
            ],
            [
                {"text": "👀 Xem media", "callback_data": f"fbp:m:{review_id}"},
                {"text": "❌ Hủy", "callback_data": f"fbp:c:{review_id}"},
            ],
        ]
    }


def parse_review_callback_data(data: str) -> Optional[Dict[str, Any]]:
    """Parse compact Telegram callback data for Facebook review buttons."""
    parts = (data or "").split(":")
    if len(parts) < 3 or parts[0] != "fbp":
        return None
    action_token = parts[1]
    review_id = parts[2]
    if not re.fullmatch(r"[0-9A-Za-z_.-]+", review_id or ""):
        return None
    if action_token == "a":
        if len(parts) != 4:
            return None
        try:
            draft_index = int(parts[3])
        except ValueError:
            return None
        if draft_index < 1 or draft_index > 3:
            return None
        return {"action": "approve", "review_id": review_id, "draft_index": draft_index}
    if action_token == "c" and len(parts) == 3:
        return {"action": "cancel", "review_id": review_id, "draft_index": None}
    if action_token == "m" and len(parts) == 3:
        return {"action": "media", "review_id": review_id, "draft_index": None}
    return None


def create_review_record(
    *,
    drafts: List[str],
    source_url: str,
    media: Optional[Dict[str, Any]],
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    source_title: str = "",
    facts: Optional[str] = None,
) -> Dict[str, Any]:
    if not drafts:
        raise ValueError("At least one draft is required")
    normalized_drafts = [str(d).strip() for d in drafts if str(d).strip()]
    if not normalized_drafts:
        raise ValueError("Drafts cannot be empty")
    digest_source = f"{source_url}|{time.time_ns()}|{'|'.join(normalized_drafts)}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:8]
    review_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{digest}"
    record = {
        "id": review_id,
        "source_url": source_url,
        "source_title": source_title,
        "facts": facts or "",
        "drafts": normalized_drafts[:3],
        "selected_media": media,
        "status": "pending_review",
        "chat_id": str(chat_id) if chat_id is not None else None,
        "thread_id": str(thread_id) if thread_id is not None else None,
        "telegram_message_id": None,
        "approved_draft_index": None,
        "published_post_id": None,
        "published_link": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "published_at": None,
        "cancelled_at": None,
    }
    _atomic_write_json(_state_path(review_id), record)
    return record


def load_review_record(review_id: str) -> Dict[str, Any]:
    path = _state_path(review_id)
    if not path.exists():
        raise FileNotFoundError(f"Review state not found: {review_id}")
    data = _read_json(path, {})
    if not isinstance(data, dict):
        raise ValueError(f"Invalid review state: {path}")
    return data


def save_review_record(record: Dict[str, Any]) -> None:
    record["updated_at"] = _now_iso()
    _atomic_write_json(_state_path(str(record["id"])), record)


async def _default_publisher(*, content: str, media_path: Optional[str], image_url: Optional[str] = None, video_url: Optional[str] = None) -> Dict[str, Any]:
    from tools.facebook_tool import facebook_post_tool

    raw = await facebook_post_tool({"content": content, "media_path": media_path, "image_url": image_url, "video_url": video_url})
    parsed = json.loads(raw)
    return parsed


async def resolve_review_action(
    review_id: str,
    action: str,
    *,
    draft_index: Optional[int] = None,
    publisher: Optional[Publisher] = None,
) -> Dict[str, Any]:
    record = load_review_record(review_id)
    status = record.get("status")

    if action == "cancel":
        if status == "published":
            return {"success": True, "status": "already_published", "record": record}
        record["status"] = "cancelled"
        record["cancelled_at"] = _now_iso()
        save_review_record(record)
        return {"success": True, "status": "cancelled", "record": record}

    if action != "approve":
        return {"success": False, "error": f"Unsupported action: {action}"}

    if status == "published":
        return {"success": True, "status": "already_published", "record": record}
    if status == "cancelled":
        return {"success": False, "status": "cancelled", "error": "Review was cancelled"}
    if status != "pending_review":
        return {"success": False, "status": status, "error": f"Review is not pending: {status}"}

    if draft_index is None:
        return {"success": False, "error": "draft_index is required for approval"}
    drafts = record.get("drafts") or []
    if draft_index < 1 or draft_index > len(drafts):
        return {"success": False, "error": f"Invalid draft index: {draft_index}"}

    media = record.get("selected_media") or {}
    media_path = media.get("path") if isinstance(media, dict) else None
    image_url = media.get("image_url") if isinstance(media, dict) else None
    video_url = media.get("video_url") if isinstance(media, dict) else None
    if media_path and not Path(media_path).exists():
        return {"success": False, "error": f"Media file not found: {media_path}"}

    publish = publisher or _default_publisher
    publish_result = await publish(content=drafts[draft_index - 1], media_path=media_path, image_url=image_url, video_url=video_url)
    if not publish_result.get("success"):
        record["last_publish_error"] = publish_result.get("error") or publish_result.get("message") or str(publish_result)
        save_review_record(record)
        return {"success": False, "status": "publish_failed", "record": record, "publish_result": publish_result}

    record["status"] = "published"
    record["approved_draft_index"] = draft_index
    record["published_post_id"] = publish_result.get("post_id") or publish_result.get("id")
    record["published_link"] = publish_result.get("link")
    record["published_at"] = _now_iso()
    save_review_record(record)
    return {"success": True, "status": "published", "record": record, "publish_result": publish_result}


def _format_media_summary(media: Optional[Dict[str, Any]]) -> str:
    if not media:
        return "Không tìm thấy media phù hợp trong thư mục."
    size_mb = (int(media.get("size_bytes") or 0) / 1_000_000)
    parts = [
        f"{media.get('filename')} ({media.get('type')}, {size_mb:.2f} MB)",
    ]
    if media.get("description"):
        parts.append(str(media["description"]))
    if media.get("tags"):
        parts.append("Tags: " + ", ".join(str(t) for t in media["tags"]))
    return "\n".join(parts)


def format_review_message(record: Dict[str, Any]) -> str:
    lines = [
        "## Bài Facebook chờ duyệt",
        f"ID: `{record['id']}`",
        f"Nguồn: {record.get('source_url') or '(không có)'}",
        "",
        "**Media đề xuất:**",
        _format_media_summary(record.get("selected_media")),
        "",
    ]
    for idx, draft in enumerate(record.get("drafts") or [], start=1):
        lines.extend([f"**Bản {idx}:**", str(draft), ""])
    lines.append("Bấm nút bên dưới để duyệt/đăng. Không bấm duyệt thì sẽ không đăng Facebook.")
    return "\n".join(lines)


async def send_review_to_telegram(
    *,
    record: Dict[str, Any],
    target: str = "telegram",
) -> Dict[str, Any]:
    from tools.send_message_tool import send_message_tool

    message = format_review_message(record)
    media = record.get("selected_media") or {}
    media_path = media.get("path") if isinstance(media, dict) else None
    if media_path:
        message = f"{message}\n\nMEDIA:{media_path}"
    live_result = await _send_review_via_live_telegram(record=record, target=target, message=message)
    if live_result is not None:
        return live_result

    raw_result = send_message_tool({"action": "send", "target": target, "message": message})
    try:
        result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
    except Exception:
        result = {"success": False, "error": str(raw_result)}
    if isinstance(result, dict) and result.get("success") and result.get("message_id"):
        record["telegram_message_id"] = str(result["message_id"])
        record["telegram_buttons"] = "unavailable: sent via send_message fallback"
        save_review_record(record)
    return result if isinstance(result, dict) else {"success": False, "error": str(result)}


async def _send_review_via_live_telegram(
    *,
    record: Dict[str, Any],
    target: str,
    message: str,
) -> Optional[Dict[str, Any]]:
    """Use the running Telegram adapter when available so inline buttons work."""
    if not target.startswith("telegram"):
        return None
    try:
        from gateway.config import Platform, load_gateway_config
        from gateway.run import _gateway_runner_ref
        from tools.send_message_tool import _parse_target_ref
    except Exception:
        return None

    runner = _gateway_runner_ref()
    if runner is None:
        return None
    adapter = runner.adapters.get(Platform.TELEGRAM)
    if adapter is None or not hasattr(adapter, "send_facebook_review"):
        return None

    chat_id = None
    thread_id = None
    target_ref = target.split(":", 1)[1].strip() if ":" in target else None
    if target_ref:
        chat_id, thread_id, _ = _parse_target_ref("telegram", target_ref)
    if not chat_id:
        config = load_gateway_config()
        home = config.get_home_channel(Platform.TELEGRAM)
        chat_id = home.chat_id if home else None
    if not chat_id:
        return None

    media = record.get("selected_media") or {}
    media_path = media.get("path") if isinstance(media, dict) else None
    result = await adapter.send_facebook_review(
        chat_id=str(chat_id),
        content=message,
        review_id=str(record["id"]),
        media_path=media_path,
        metadata={"thread_id": thread_id} if thread_id else None,
    )
    if result.success:
        record["telegram_message_id"] = str(result.message_id) if result.message_id else None
        record["telegram_buttons"] = "enabled"
        save_review_record(record)
        return {"success": True, "message_id": result.message_id, "buttons": True}
    return {"success": False, "error": result.error or "Telegram send_facebook_review failed"}


def _basic_http_extract_sync(url: str) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        content_type = response.headers.get("content-type", "")
        charset_match = re.search(r"charset=([^;]+)", content_type, re.I)
        charset = (charset_match.group(1).strip() if charset_match else "utf-8") or "utf-8"
        raw = response.read(2_000_000)
    html_text = raw.decode(charset, errors="replace")
    title_match = re.search(r"<title[^>]*>([\s\S]*?)</title>", html_text, re.I)
    title = html.unescape(re.sub(r"\s+", " ", title_match.group(1)).strip()) if title_match else ""
    text = re.sub(r"<script[\s\S]*?</script>", " ", html_text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<noscript[\s\S]*?</noscript>", " ", text, flags=re.I)
    text = re.sub(r"</(p|div|section|article|h[1-6]|li|br)>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return {"title": title, "content": text, "fallback_source": "basic_http"}


async def _basic_http_extract(url: str) -> Dict[str, Any]:
    try:
        return await asyncio.to_thread(_basic_http_extract_sync, url)
    except Exception as exc:
        return {"title": "", "content": "", "error": f"Basic HTTP extraction failed: {exc}"}


async def _crawl4ai_extract(url: str) -> Dict[str, Any]:
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    except Exception as exc:
        return {"title": "", "content": "", "error": f"Crawl4AI unavailable: {exc}"}

    try:
        browser_config = BrowserConfig(headless=True)
        crawler_config = CrawlerRunConfig(
            page_timeout=60000,
            remove_overlay_elements=True,
            excluded_tags=["nav", "footer", "aside", "script", "style"],
        )
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=crawler_config)
        if not getattr(result, "success", False):
            error = getattr(result, "error_message", None) or getattr(result, "error", None) or "Crawl4AI extraction failed"
            return {"title": "", "content": "", "error": str(error)}
        markdown = getattr(result, "markdown", "") or ""
        content = getattr(markdown, "fit_markdown", None) or getattr(markdown, "raw_markdown", None) or str(markdown)
        metadata = getattr(result, "metadata", {}) or {}
        title = metadata.get("title") or ""
        return {"title": title, "content": content.strip(), "fallback_source": "crawl4ai"}
    except Exception as exc:
        return {"title": "", "content": "", "error": f"Crawl4AI extraction failed: {exc}"}


async def _crawl_fallback(url: str) -> Dict[str, Any]:
    crawl4ai_result = await _crawl4ai_extract(url)
    if not crawl4ai_result.get("error") and crawl4ai_result.get("content"):
        return crawl4ai_result
    basic_result = await _basic_http_extract(url)
    if not basic_result.get("error") and basic_result.get("content"):
        return basic_result
    return crawl4ai_result if crawl4ai_result.get("error") else basic_result


async def _extract_url(url: str) -> Dict[str, Any]:
    try:
        from tools.web_tools import web_extract_tool

        raw = await web_extract_tool([url], "markdown")
    except Exception as exc:
        fallback = await _crawl_fallback(url)
        if not fallback.get("error") and fallback.get("content"):
            return fallback
        return {"title": "", "content": "", "error": str(exc)}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {"title": "", "content": raw}
    results = parsed.get("results") or parsed.get("data", {}).get("results") or []
    first = results[0] if results else {}
    error = first.get("error")
    content = first.get("content") or first.get("text") or ""
    if (error and not content) or not content.strip():
        fallback = await _crawl_fallback(url)
        if not fallback.get("error") and fallback.get("content"):
            return fallback
    return {
        "title": first.get("title") or "",
        "content": content,
        "error": error,
    }


def _fallback_drafts(title: str, content: str, source_url: str) -> List[str]:
    text = re.sub(r"\s+", " ", content or "").strip()
    excerpt = text[:900] + ("..." if len(text) > 900 else "")
    title = title or "Chủ đề hôm nay"
    return [
        f"{title}\n\n{excerpt}\n\nXem thêm: {source_url}",
        f"Bạn đang quan tâm đến {title.lower()}?\n\nMột vài ý chính đáng chú ý: {excerpt}\n\nNguồn: {source_url}",
        f"Góc chia sẻ hôm nay: {title}\n\n{excerpt}\n\nNếu bạn thấy hữu ích, hãy lưu lại để tham khảo nhé.",
    ]


async def _generate_drafts(
    title: str,
    content: str,
    source_url: str,
    selected_media: Optional[Dict[str, Any]] = None,
) -> List[str]:
    try:
        from agent.auxiliary_client import (
            async_call_llm,
            extract_content_or_reasoning,
            get_async_text_auxiliary_client,
            get_auxiliary_extra_body,
        )

        client, model = get_async_text_auxiliary_client("web_extract")
        if not client or not model:
            return _fallback_drafts(title, content, source_url)
        compact = (content or "")[:6000]
        media_context = "Không có media phù hợp được chọn. Viết bài text-only, không nhắc đến ảnh."
        if selected_media:
            media_context = json.dumps(
                {
                    "filename": selected_media.get("filename"),
                    "description": selected_media.get("description"),
                    "tags": selected_media.get("tags"),
                    "type": selected_media.get("type"),
                },
                ensure_ascii=False,
            )
        prompt = f"""Viết đúng 3 bản nháp bài Facebook tiếng Việt dựa trên nội dung web và media đã chọn bên dưới.
Yêu cầu: tự nhiên, không bịa claim ngoài nguồn, mỗi bản 80-160 từ, phù hợp đăng Facebook.
Nếu có media đã chọn, nội dung bài phải khớp rõ với tag/mô tả media đó. Nếu không có media, viết bài text-only và không nhắc đến ảnh.
Trả về JSON array gồm 3 string, không markdown ngoài JSON.

Nguồn: {source_url}
Tiêu đề: {title}
Media đã chọn:
{media_context}
Nội dung rút gọn:
{compact}
"""
        response = await async_call_llm(
            task="web_extract",
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1800,
            extra_body=get_auxiliary_extra_body() or {},
        )
        text = extract_content_or_reasoning(response) or ""
        match = re.search(r"\[[\s\S]*\]", text)
        drafts = json.loads(match.group(0) if match else text)
        if isinstance(drafts, list) and len(drafts) >= 3:
            return [str(d).strip() for d in drafts[:3] if str(d).strip()]
    except Exception:
        logger.warning("Falling back to deterministic Facebook drafts", exc_info=True)
    return _fallback_drafts(title, content, source_url)


async def create_facebook_review_tool(args: Dict[str, Any], **_: Any) -> str:
    try:
        source_url = args.get("source_url")
        if not source_url:
            source_cfg = _read_json(SOURCE_CONFIG, {})
            source_url = source_cfg.get("source_url")
        if not source_url:
            return tool_error("Missing source_url and no source config found")

        media_dir = Path(args.get("media_dir") or DEFAULT_MEDIA_DIR)
        extracted = await _extract_url(str(source_url))
        if extracted.get("error"):
            return tool_error(f"Source extraction failed: {extracted['error']}")
        content = extracted.get("content") or ""
        title = extracted.get("title") or ""
        media = select_media(discover_media(media_dir), hint=f"{title} {content[:1000]}")
        drafts = args.get("drafts")
        if not isinstance(drafts, list) or not drafts:
            drafts = await _generate_drafts(title, content, str(source_url), selected_media=media)

        record = create_review_record(
            drafts=[str(d) for d in drafts[:3]],
            source_url=str(source_url),
            source_title=title,
            facts=content[:2000],
            media=media,
            chat_id=args.get("chat_id"),
            thread_id=args.get("thread_id"),
        )
        target = args.get("target") or "telegram"
        send_result = await send_review_to_telegram(record=record, target=target)
        return tool_result(success=True, review_id=record["id"], record=record, telegram=send_result)
    except Exception as exc:
        logger.exception("create_facebook_review failed")
        return tool_error(str(exc))


async def facebook_review_action_tool(args: Dict[str, Any], **_: Any) -> str:
    review_id = args.get("review_id")
    action = args.get("action")
    if not review_id or not action:
        return tool_error("review_id and action are required")
    result = await resolve_review_action(
        str(review_id),
        str(action),
        draft_index=args.get("draft_index"),
    )
    return json.dumps(result, ensure_ascii=False)


CREATE_REVIEW_SCHEMA = {
    "name": "facebook_create_review",
    "description": "Create Facebook post drafts from a source URL, attach local media from the profile's automations/FacebookMedia directory, and send to Telegram for button approval. Does not publish.",
    "parameters": {
        "type": "object",
        "properties": {
            "source_url": {"type": "string", "description": "Optional source URL. Defaults to facebook_daily_source.json."},
            "media_dir": {"type": "string", "description": "Local media directory. Defaults to the Hermes profile's automations/FacebookMedia directory."},
            "target": {"type": "string", "description": "Telegram delivery target, default telegram."},
            "drafts": {"type": "array", "items": {"type": "string"}, "description": "Optional prebuilt drafts."},
            "chat_id": {"type": "string"},
            "thread_id": {"type": "string"},
        },
        "required": [],
    },
}

REVIEW_ACTION_SCHEMA = {
    "name": "facebook_review_action",
    "description": "Resolve a pending Facebook review: approve/publish a selected draft or cancel it.",
    "parameters": {
        "type": "object",
        "properties": {
            "review_id": {"type": "string"},
            "action": {"type": "string", "enum": ["approve", "cancel"]},
            "draft_index": {"type": "integer", "minimum": 1, "maximum": 3},
        },
        "required": ["review_id", "action"],
    },
}

registry.register(
    name="facebook_create_review",
    toolset="facebook",
    schema=CREATE_REVIEW_SCHEMA,
    handler=create_facebook_review_tool,
    is_async=True,
    emoji="📝",
)
registry.register(
    name="facebook_review_action",
    toolset="facebook",
    schema=REVIEW_ACTION_SCHEMA,
    handler=facebook_review_action_tool,
    is_async=True,
    emoji="✅",
)
