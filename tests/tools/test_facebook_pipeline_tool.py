"""Tests for Facebook draft review pipeline helpers."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def test_discovers_media_from_metadata_without_reading_pixels(tmp_path):
    media_dir = tmp_path / "FacebookMedia"
    media_dir.mkdir()
    image = media_dir / "2026-05-19-monstera-care-01.jpg"
    image.write_bytes(b"fake-jpeg-bytes")
    (media_dir / "media.json").write_text(
        json.dumps({
            "media": [
                {
                    "file": "2026-05-19-monstera-care-01.jpg",
                    "description": "Ảnh lá monstera xanh khỏe",
                    "tags": ["monstera", "care"],
                }
            ]
        }),
        encoding="utf-8",
    )

    from tools.facebook_pipeline_tool import discover_media

    discovered = discover_media(media_dir)

    assert len(discovered) == 1
    assert discovered[0]["path"] == str(image)
    assert discovered[0]["description"] == "Ảnh lá monstera xanh khỏe"
    assert discovered[0]["type"] == "image"
    assert discovered[0]["size_bytes"] == len(b"fake-jpeg-bytes")
    assert "content" not in discovered[0]
    assert "bytes" not in discovered[0]


def test_select_media_prefers_matching_tag_then_newest(tmp_path):
    media_dir = tmp_path / "FacebookMedia"
    media_dir.mkdir()
    old = media_dir / "2026-05-18-monstera-care-01.jpg"
    new_unmatched = media_dir / "2026-05-19-snake-plant-01.jpg"
    old.write_bytes(b"old")
    new_unmatched.write_bytes(b"new")
    (media_dir / "media.json").write_text(
        json.dumps({
            "media": [
                {"file": old.name, "tags": ["monstera", "care"]},
                {"file": new_unmatched.name, "tags": ["snake-plant"]},
            ]
        }),
        encoding="utf-8",
    )

    from tools.facebook_pipeline_tool import discover_media, select_media

    selected = select_media(discover_media(media_dir), hint="Bài viết về chăm sóc monstera")

    assert selected["filename"] == old.name


def test_select_media_does_not_fallback_to_newest_when_tags_do_not_match(tmp_path):
    media_dir = tmp_path / "FacebookMedia"
    media_dir.mkdir()
    unrelated = media_dir / "2026-05-20-ngua-van-01.jpg"
    unrelated.write_bytes(b"new")
    (media_dir / "media.json").write_text(
        json.dumps({"media": [{"file": unrelated.name, "tags": ["ngựa vằn", "calathea"]}]}),
        encoding="utf-8",
    )

    from tools.facebook_pipeline_tool import discover_media, select_media

    selected = select_media(discover_media(media_dir), hint="Bài viết về cây kim tiền và tài lộc")

    assert selected is None


def test_extract_url_falls_back_to_crawl4ai_when_web_tools_unconfigured(monkeypatch):
    from tools import facebook_pipeline_tool as fp
    from tools import web_tools

    async def fake_web_extract(_urls, _format):
        return json.dumps({
            "results": [
                {"error": "Web tools are not configured. Set FIRECRAWL_API_KEY."}
            ]
        })

    async def fake_crawl4ai(_url):
        return {"title": "Crawl4AI title", "content": "Crawl4AI content", "fallback_source": "crawl4ai"}

    async def fake_plain_http(_url):
        return {"title": "", "content": "", "error": "plain HTTP should not run before Crawl4AI"}

    monkeypatch.setattr(web_tools, "web_extract_tool", fake_web_extract)
    monkeypatch.setattr(fp, "_crawl4ai_extract", fake_crawl4ai, raising=False)
    monkeypatch.setattr(fp, "_basic_http_extract", fake_plain_http)

    result = asyncio.run(fp._extract_url("https://example.com/article"))

    assert result["title"] == "Crawl4AI title"
    assert result["content"] == "Crawl4AI content"
    assert result["fallback_source"] == "crawl4ai"
    assert "error" not in result


def test_extract_url_falls_back_to_crawl4ai_when_web_tools_return_empty_content(monkeypatch):
    from tools import facebook_pipeline_tool as fp
    from tools import web_tools

    async def fake_web_extract(_urls, _format):
        return json.dumps({"results": [{"title": "", "content": ""}]})

    async def fake_crawl4ai(_url):
        return {"title": "Crawl4AI title", "content": "Crawl4AI content", "fallback_source": "crawl4ai"}

    monkeypatch.setattr(web_tools, "web_extract_tool", fake_web_extract)
    monkeypatch.setattr(fp, "_crawl4ai_extract", fake_crawl4ai, raising=False)

    result = asyncio.run(fp._extract_url("https://example.com/article"))

    assert result["title"] == "Crawl4AI title"
    assert result["content"] == "Crawl4AI content"
    assert result["fallback_source"] == "crawl4ai"
    assert "error" not in result


def test_extract_url_falls_back_to_plain_http_when_crawl4ai_unavailable(monkeypatch):
    from tools import facebook_pipeline_tool as fp
    from tools import web_tools

    async def fake_web_extract(_urls, _format):
        return json.dumps({
            "results": [
                {"error": "Web tools are not configured. Set FIRECRAWL_API_KEY."}
            ]
        })

    async def fake_crawl4ai(_url):
        return {"title": "", "content": "", "error": "Crawl4AI not installed"}

    async def fake_plain_http(_url):
        return {"title": "Fallback title", "content": "Fallback content", "fallback_source": "basic_http"}

    monkeypatch.setattr(web_tools, "web_extract_tool", fake_web_extract)
    monkeypatch.setattr(fp, "_crawl4ai_extract", fake_crawl4ai, raising=False)
    monkeypatch.setattr(fp, "_basic_http_extract", fake_plain_http)

    result = asyncio.run(fp._extract_url("https://example.com/article"))

    assert result["title"] == "Fallback title"
    assert result["content"] == "Fallback content"
    assert result["fallback_source"] == "basic_http"
    assert "error" not in result



def test_create_review_generates_drafts_after_selecting_matching_media(tmp_path, monkeypatch):
    from tools import facebook_pipeline_tool as fp

    media_dir = tmp_path / "FacebookMedia"
    media_dir.mkdir()
    image = media_dir / "2026-05-20-kim-tien-01.jpg"
    image.write_bytes(b"img")
    (media_dir / "media.json").write_text(
        json.dumps({"media": [{"file": image.name, "tags": ["kim tiền", "zamioculcas"]}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(fp, "STATE_DIR", tmp_path / "state")

    async def fake_extract(_url):
        return {"title": "Cây kim tiền", "content": "Cây kim tiền hợp văn phòng và mang ý nghĩa tài lộc."}

    seen = {}

    async def fake_generate(title, content, source_url, selected_media=None):
        seen["selected_media"] = selected_media
        return ["draft 1 về kim tiền", "draft 2 về kim tiền", "draft 3 về kim tiền"]

    async def fake_send(*, record, target="telegram"):
        seen["record"] = record
        return {"success": True, "message_id": "1", "buttons": True}

    monkeypatch.setattr(fp, "_extract_url", fake_extract)
    monkeypatch.setattr(fp, "_generate_drafts", fake_generate)
    monkeypatch.setattr(fp, "send_review_to_telegram", fake_send)

    raw = asyncio.run(fp.create_facebook_review_tool({"source_url": "https://example.com/kim-tien", "media_dir": str(media_dir)}))
    result = json.loads(raw)

    assert result["success"] is True
    assert seen["selected_media"]["filename"] == image.name
    assert seen["record"]["selected_media"]["filename"] == image.name


def test_review_state_approval_is_idempotent(tmp_path, monkeypatch):
    from tools import facebook_pipeline_tool as fp

    monkeypatch.setattr(fp, "STATE_DIR", tmp_path / "state")
    media = tmp_path / "image.jpg"
    media.write_bytes(b"img")
    record = fp.create_review_record(
        drafts=["draft 1", "draft 2", "draft 3"],
        source_url="https://example.com/article",
        media={"path": str(media), "filename": media.name, "type": "image"},
        chat_id="123",
    )

    calls = []

    async def fake_publisher(*, content, media_path, image_url=None, video_url=None):
        calls.append({
            "content": content,
            "media_path": media_path,
            "image_url": image_url,
            "video_url": video_url,
        })
        return {"success": True, "post_id": "page_post", "link": "https://facebook.com/page_post"}

    first = asyncio.run(
        fp.resolve_review_action(record["id"], "approve", draft_index=2, publisher=fake_publisher)
    )
    second = asyncio.run(
        fp.resolve_review_action(record["id"], "approve", draft_index=2, publisher=fake_publisher)
    )

    assert first["success"] is True
    assert first["status"] == "published"
    assert second["success"] is True
    assert second["status"] == "already_published"
    assert len(calls) == 1
    assert calls[0] == {
        "content": "draft 2",
        "media_path": str(media),
        "image_url": None,
        "video_url": None,
    }


def test_default_media_dir_uses_hermes_profile_home():
    from tools.facebook_pipeline_tool import CREATE_REVIEW_SCHEMA, DEFAULT_MEDIA_DIR

    assert DEFAULT_MEDIA_DIR.parts[-2:] == ("automations", "FacebookMedia")
    schema_text = json.dumps(CREATE_REVIEW_SCHEMA, ensure_ascii=False)
    assert "/Desktop/FacebookMedia" not in schema_text


def test_build_telegram_keyboard_has_short_callback_data():
    from tools.facebook_pipeline_tool import build_review_keyboard

    keyboard = build_review_keyboard("20260519-abc123")

    payload = keyboard["inline_keyboard"]
    callback_values = [button["callback_data"] for row in payload for button in row]
    assert "fbp:a:20260519-abc123:1" in callback_values
    assert "fbp:a:20260519-abc123:2" in callback_values
    assert "fbp:a:20260519-abc123:3" in callback_values
    assert "fbp:m:20260519-abc123" in callback_values
    assert "fbp:c:20260519-abc123" in callback_values
    assert all(len(value.encode("utf-8")) <= 64 for value in callback_values)


def test_parses_short_facebook_callback_data():
    from tools.facebook_pipeline_tool import parse_review_callback_data

    approve = parse_review_callback_data("fbp:a:20260519-abc123:2")
    cancel = parse_review_callback_data("fbp:c:20260519-abc123")
    media = parse_review_callback_data("fbp:m:20260519-abc123")

    assert approve == {"action": "approve", "review_id": "20260519-abc123", "draft_index": 2}
    assert cancel == {"action": "cancel", "review_id": "20260519-abc123", "draft_index": None}
    assert media == {"action": "media", "review_id": "20260519-abc123", "draft_index": None}
    assert parse_review_callback_data("fbp:a:bad") is None
    assert parse_review_callback_data("other") is None
