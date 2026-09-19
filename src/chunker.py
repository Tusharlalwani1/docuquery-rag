import os
import re

def load_documents_from_folder(folder_path: str) -> str:
    combined_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith((".txt", ".md")):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                combined_text += f"\n\n{f.read()}"
    return combined_text

def create_sentence_chunks(text: str, chunk_size: int = 3, overlap: int = 1) -> list[str]:
    """
    Splits text by Markdown paragraphs and tables using double newlines,
    preventing tables and lists from being torn apart by periods.
    """
    # Split by double newlines to separate paragraphs naturally
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    chunks = []
    
    if len(paragraphs) <= chunk_size:
        return ["\n\n".join(paragraphs)]
        
    i = 0
    while i < len(paragraphs):
        # Join paragraphs back together maintaining original table/list formatting
        chunk = "\n\n".join(paragraphs[i : i + chunk_size])
        chunks.append(chunk)
        i += max(1, chunk_size - overlap)
        
    return chunks