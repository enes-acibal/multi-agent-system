"""
File reader tool — lets the agent read local files.

This is deliberately simple. The agent can read text-based files
(txt, md, py, json, csv, etc.). For PDFs, you'd add a PDF parser.

SECURITY NOTE: In production, you MUST restrict file paths to a safe
directory (sandbox). Never let an agent read arbitrary system files.
"""
import os


SAFE_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def read_file(file_path: str) -> str:
    """
    Read a text file from the safe data directory.
    
    Args:
        file_path: Relative path within the data/ directory.
        
    Returns:
        The file contents as a string, or an error message.
    """
    # Resolve the full path and ensure it stays within our safe directory
    full_path = os.path.normpath(os.path.join(SAFE_DIRECTORY, file_path))
    safe_dir = os.path.normpath(SAFE_DIRECTORY)
    
    if not full_path.startswith(safe_dir):
        return "Error: Access denied. Path is outside the allowed directory."
    
    if not os.path.exists(full_path):
        return f"Error: File not found: {file_path}"
    
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Truncate very large files to prevent token overflow
        if len(content) > 10000:
            content = content[:10000] + "\n\n... [truncated — file too large]"
        
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


FILE_READER_TOOL = {
    "name": "read_file",
    "description": (
        "Read the contents of a text file from the project's data directory. "
        "Supports .txt, .md, .py, .json, .csv, and other text formats. "
        "Use this to examine documents, code files, or data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": (
                    "Relative path to the file within the data/ directory. "
                    "Example: 'documents/report.md'"
                )
            }
        },
        "required": ["file_path"]
    }
}