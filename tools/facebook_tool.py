#!/usr/bin/env python3
import json
import logging
import os
import httpx
from typing import Dict, Any, Optional, List
from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)

def get_fb_config():
    """Read Facebook config from environment variables."""
    from hermes_cli.config import get_env_value
    return {
        "page_id": get_env_value("FACEBOOK_PAGE_ID"),
        "access_token": get_env_value("FACEBOOK_PAGE_ACCESS_TOKEN"),
    }

def check_facebook_requirements() -> bool:
    """Check if Facebook credentials are set."""
    config = get_fb_config()
    return bool(config["page_id"] and config["access_token"])

async def facebook_post_tool(args: Dict[str, Any]) -> str:
    """
    Publish content (text, images, or video) to a Facebook Page.
    Returns the post ID or URL on success.
    """
    config = get_fb_config()
    page_id = config["page_id"]
    token = config["access_token"]
    
    content = args.get("content", "")
    image_url = args.get("image_url")
    video_url = args.get("video_url")
    media_path = args.get("media_path") # Local path for upload
    
    base_url = f"https://graph.facebook.com/v21.0/{page_id}"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Case 1: Video Post
            if video_url or (media_path and media_path.lower().endswith(('.mp4', '.mov', '.avi'))):
                url = f"{base_url}/videos"
                data = {"description": content, "access_token": token}
                if video_url:
                    data["file_url"] = video_url
                    res = await client.post(url, data=data)
                else:
                    # Upload local file
                    files = {'file': open(media_path, 'rb')}
                    res = await client.post(url, data=data, files=files)
            
            # Case 2: Image Post
            elif image_url or (media_path and media_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))):
                url = f"{base_url}/photos"
                data = {"caption": content, "access_token": token}
                if image_url:
                    data["url"] = image_url
                    res = await client.post(url, data=data)
                else:
                    # Upload local file
                    files = {'source': open(media_path, 'rb')}
                    res = await client.post(url, data=data, files=files)
            
            # Case 3: Text Post
            else:
                url = f"{base_url}/feed"
                data = {"message": content, "access_token": token}
                res = await client.post(url, data=data)
                
            res.raise_for_status()
            result_data = res.json()
            # Return post ID and a direct link if possible
            post_id = result_data.get("id") or result_data.get("post_id")
            return tool_result(
                success=True,
                post_id=post_id,
                message="Successfully published to Facebook.",
                link=f"https://facebook.com/{post_id}"
            )
            
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("error", {}).get("message", str(e))
            return tool_error(f"Facebook API error: {error_detail}")
        except Exception as e:
            return tool_error(f"Unexpected error: {str(e)}")

FACEBOOK_POST_SCHEMA = {
    "name": "facebook_post",
    "description": "Publish a post to a Facebook Page with optional text, image, or video.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The text content or caption of the post."
            },
            "image_url": {
                "type": "string",
                "description": "Public URL of an image to post."
            },
            "video_url": {
                "type": "string",
                "description": "Public URL of a video to post."
            },
            "media_path": {
                "type": "string",
                "description": "Local absolute path to an image or video file to upload."
            }
        },
        "required": ["content"]
    }
}

registry.register(
    name="facebook_post",
    toolset="facebook",
    schema=FACEBOOK_POST_SCHEMA,
    handler=facebook_post_tool,
    check_fn=check_facebook_requirements,
    requires_env=["FACEBOOK_PAGE_ID", "FACEBOOK_PAGE_ACCESS_TOKEN"],
    is_async=True,
    emoji="📘"
)
