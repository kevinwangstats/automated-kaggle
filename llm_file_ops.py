import os
from pathlib import Path
from openai import OpenAI
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
        
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    except Exception as e:
        log_error("Failed to initialize OpenAI client for file extraction.", e)
        return None
    
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            continue
            
        log_info(f"Uploading {fp} to LLM File API for extraction...")
        try:
            # 1. Upload
            file_object = client.files.create(
                file=path,
                purpose="file-extract"
            )
            
            # 2. Extract
            file_content = client.files.content(file_id=file_object.id).text
            
            # 3. Format message
            messages.append({
                "role": "system",
                "content": f"--- {fp} ---\n{file_content}"
            })
            
            # Delete file to avoid hitting limits
            try:
                client.files.delete(file_object.id)
            except Exception as e:
                log_error(f"Failed to clean up file {file_object.id}", e)
            
        except Exception as e:
            log_error(f"Failed to process {fp} with LLM File API", e)
            return None # Fallback to local on any failure
            
    return messages
