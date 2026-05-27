import os
import requests
from pathlib import Path
from logger import log_info, log_error

def get_llm_file_messages(file_paths: list, api_key: str, base_url: str = "https://api.openai.com/v1") -> list:
    """
    Uploads files to an LLM File API, extracts their contents, and formats them
    as system messages. This avoids local truncation and uses native parsing
    and context caching capabilities.
    """
    messages = []
    if not api_key:
        log_info("API key not found. Falling back to local file reading.")
        return None
        
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Strip trailing slashes from base_url for consistent endpoint construction
    base_url = base_url.rstrip("/")
    
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            continue
            
        log_info(f"Uploading {fp} to LLM File API for extraction...")
        try:
            # 1. Upload
            url_upload = f"{base_url}/files"
            with open(path, "rb") as f:
                files = {
                    "file": (path.name, f, "application/octet-stream"),
                    "purpose": (None, "file-extract")
                }
                resp = requests.post(url_upload, headers=headers, files=files)
            resp.raise_for_status()
            file_id = resp.json()["id"]
            
            # 2. Extract
            url_content = f"{base_url}/files/{file_id}/content"
            content_resp = requests.get(url_content, headers=headers)
            content_resp.raise_for_status()
            
            # API returns {"content": "..."} or raw text depending on exact behavior,
            # but standard is JSON with 'content'. Let's handle both.
            try:
                json_data = content_resp.json()
                file_content = json_data.get("content", content_resp.text)
            except Exception:
                file_content = content_resp.text
            
            # 3. Format message
            messages.append({
                "role": "system",
                "content": f"--- {fp} ---\n{file_content}"
            })
            
            # Delete file to avoid hitting limits
            try:
                requests.delete(f"{base_url}/files/{file_id}", headers=headers)
            except Exception as e:
                log_error(f"Failed to clean up file {file_id}", e)
            
        except Exception as e:
            log_error(f"Failed to process {fp} with LLM File API", e)
            return None # Fallback to local on any failure
            
    return messages
