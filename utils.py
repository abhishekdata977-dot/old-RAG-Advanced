"""
RAG Prototype - Utility Functions Module

This module contains all core functionality for the RAG (Retrieval-Augmented Generation) pipeline:
- Document loading and processing
- Text chunking and embedding generation
- Vector storage and similarity search
- Google Gemini integration for answer generation

Author: RAG Prototype Team
Date: 2024
"""

import os
import json
import pickle
import numpy as np
import hashlib
import uuid
import logging
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import re
import unicodedata
import difflib
from collections import Counter
from rank_bm25 import BM25Okapi
# Optional audio playback import
try:
    from playsound import playsound
    AUDIO_PLAYBACK_AVAILABLE = True
except ImportError:
    logging.warning("playsound module not available. Audio playback features will be disabled.")
    AUDIO_PLAYBACK_AVAILABLE = False
    playsound = None

stop_audio_flag = False
# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Directory paths for data storage
EMBEDDINGS_DIR = "embeddings"
DATA_DIR = "data"

# Load environment variables from .env file for security
load_dotenv()

# Load Gemini API key from environment variables (security best practice)
# Support both GEMINI_API_KEY and GOOGLE_API_KEY for compatibility
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables")
    print("Please create a .env file with your Google Gemini API key")

# Configure Google Generative AI with API key
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: Google Generative AI not configured due to missing API key")

# Initialize SentenceTransformer model with error handling
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ SentenceTransformer model loaded successfully")
except Exception as e:
    print(f"❌ Error loading SentenceTransformer model: {e}")
    embedding_model = None

# Initialize CrossEncoder model with error handling
try:
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("✅ CrossEncoder reranker model loaded successfully")
except Exception as e:
    print(f"❌ Error loading CrossEncoder reranker model: {e}")
    reranker_model = None

# =============================================================================
# TEXT NORMALIZATION FUNCTION (NEW)
# =============================================================================
def normalize_text(text: str) -> str:
    """
    Normalize text for improved embedding accuracy.
    - Lowercase
    - Unicode normalization
    - Remove extra whitespace
    - Remove non-informative characters (except basic punctuation)
    """
    if not text:
        return ""
    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    # Lowercase
    text = text.lower()
    # Remove non-informative characters (keep basic punctuation)
    text = re.sub(r"[^\w\s.,;:!?()\[\]'-]", "", text)
    # Collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# =============================================================================

def extract_pdf_structure(pages_text: list) -> list:
    """
    Given a list of (page_num, page_text), return a list of dicts with page_text, chapter_title, chapter_number, section_title.
    Uses regex to detect chapter/section headings (e.g., '1 Why Machine Learning Strategy').
    """
    structure = []
    current_chapter = None
    current_chapter_number = None
    current_section = None
    chapter_pattern = re.compile(r"^(\d+)\s+(.+)", re.MULTILINE)
    section_pattern = re.compile(r"^(\d+\.\d+)\s+(.+)", re.MULTILINE)
    for page_num, page_text in pages_text:
        chapter_found = chapter_pattern.findall(page_text)
        section_found = section_pattern.findall(page_text)
        # Use the last match on the page (if multiple headings)
        if chapter_found:
            current_chapter_number = chapter_found[-1][0]
            current_chapter = f"{chapter_found[-1][0]} {chapter_found[-1][1].strip()}"
        if section_found:
            current_section = f"{section_found[-1][0]} {section_found[-1][1].strip()}"
        structure.append({
            'page_num': page_num,
            'page_text': page_text,
            'chapter_title': current_chapter,
            'chapter_number': current_chapter_number,
            'section_title': current_section
        })
    return structure

def load_document(file_path: str) -> Tuple[Optional[str], Optional[list], Optional[list], Optional[str]]:
    """
    Loads text content from PDF or TXT files with comprehensive error handling.
    Returns (full_text, structured_pages, toc_data, error)
    structured_pages is a list of dicts with page_text, chapter_title, section_title.
    toc_data is a list of dicts: [{'number': '1', 'title': 'Overview', 'page': 3}, ...]
    """
    try:
        if not os.path.exists(file_path):
            return None, None, None, f"File not found: {file_path}"
        if not os.access(file_path, os.R_OK):
            return None, None, None, f"Permission denied: Cannot read {file_path}"
        if file_path.lower().endswith('.pdf'):
            text = ""
            pages_text = []
            toc_data = []
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    if pdf_reader.is_encrypted:
                        return None, None, None, f"PDF is encrypted: {file_path}"
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                                pages_text.append((page_num, page_text))
                        except Exception as e:
                            print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                            continue
                # --- Robust TOC Extraction (Enhanced) ---
                toc_found = False
                toc_lines = []
                # Try to find explicit TOC first
                for page_num, page_text in pages_text[:15]:
                    if not toc_found and re.search(r"table of contents|contents", page_text, re.IGNORECASE):
                        toc_found = True
                        lines = page_text.splitlines()
                        start_idx = None
                        for i, line in enumerate(lines):
                            if re.search(r"table of contents|contents", line, re.IGNORECASE):
                                start_idx = i + 1
                                break
                        if start_idx is not None:
                            toc_lines.extend(lines[start_idx:])
                        continue
                    elif toc_found:
                        # Heuristic: stop at first real section/chapter
                        if re.search(r"introduction|overview|chapter 1|^1 ", page_text, re.IGNORECASE):
                            break
                        toc_lines.extend(page_text.splitlines())
                # If no explicit TOC, look for a numbered list TOC in first 3 pages
                if not toc_lines:
                    for page_num, page_text in pages_text[:3]:
                        lines = page_text.splitlines()
                        numbered_lines = [l for l in lines if re.match(r"^\s*\d+(?:\.\d+)*[ .\t]+.+\s+\d+\s*$", l)]
                        if len(numbered_lines) >= 3:  # Heuristic: at least 3 TOC-like lines
                            toc_lines = numbered_lines
                            break
                # Enhanced TOC regex: supports dotted lines, variable whitespace, section numbers
                toc_entry_pattern = re.compile(r'^\s*(\d+(?:\.\d+)*)(?:\s+|\s*[.·•]+\s*)+(.+?)(?:\s*[.·•]+\s*)+(\d+)\s*$')
                for line in toc_lines:
                    match = toc_entry_pattern.match(line.strip())
                    if match:
                        number, title, page = match.groups()
                        toc_data.append({'number': number, 'title': title.strip(), 'page': int(page)})
                # Fallback: try to infer TOC if not found
                if not toc_data:
                    inferred_toc = []
                    heading_pattern = re.compile(r'^\s*(\d+(?:\.\d+)*)(?:\s+)([A-Za-z].{3,})', re.MULTILINE)
                    for page_num, page_text in pages_text[:20]:
                        for match in heading_pattern.finditer(page_text):
                            number, title = match.groups()
                            inferred_toc.append({'number': number, 'title': title.strip(), 'page': page_num + 1})
                    toc_data = inferred_toc if inferred_toc else None
                structured_pages = extract_pdf_structure(pages_text)
                return normalize_text(text.strip()), structured_pages, toc_data, None
            except Exception as e:
                return None, None, None, f"Error reading PDF {file_path}: {e}"
        elif file_path.lower().endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                return normalize_text(text.strip()), None, None, None
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as file:
                        text = file.read()
                    return normalize_text(text.strip()), None, None, None
                except Exception as e:
                    return None, None, None, f"Error reading TXT file {file_path}: {e}"
            except Exception as e:
                return None, None, None, f"Error reading TXT file {file_path}: {e}"
        else:
            return None, None, None, f"Unsupported file type: {file_path}. Supported formats: PDF, TXT"
    except Exception as e:
        return None, None, None, f"Unexpected error reading file {file_path}: {e}"

def parse_table_from_text(text):
    """
    Parse table-like text into a list of dicts (if headers detected) or lists.
    Supports multi-line headers and merged cell filling.
    Returns (table_data, headers) or (None, None) if not a table.
    """
    import re
    lines = [l for l in text.splitlines() if l.strip()]
    # Detect delimiter (|, tab, or multiple spaces)
    delimiter = None
    if any('|' in l for l in lines):
        delimiter = '|'
    elif any('\t' in l for l in lines):
        delimiter = '\t'
    elif any(re.search(r'\s{2,}', l) for l in lines):
        delimiter = '  '
    if not delimiter:
        return None, None
    # Split lines into columns
    split_lines = [re.split(r'\s*\|\s*' if delimiter == '|' else delimiter, l.strip()) for l in lines]
    split_lines = [[c for c in row if c.strip()] for row in split_lines]
    # Multi-line header detection: first 2 lines are headers if both are mostly text and similar col count
    header_rows = 1
    if len(split_lines) > 1 and abs(len(split_lines[0]) - len(split_lines[1])) <= 1:
        if all(any(c.isalpha() for c in cell) for cell in split_lines[0]) and all(any(c.isalpha() for c in cell) for cell in split_lines[1]):
            header_rows = 2
    headers = []
    if header_rows == 2:
        for h1, h2 in zip(split_lines[0], split_lines[1]):
            headers.append(f"{h1.strip()} {h2.strip()}")
    elif header_rows == 1:
        headers = split_lines[0]
    data_rows = split_lines[header_rows:]
    # Fill down merged cells (empty cells get value from above)
    for row_idx, row in enumerate(data_rows):
        for col_idx, cell in enumerate(row):
            if cell.strip() == '' and row_idx > 0:
                row[col_idx] = data_rows[row_idx-1][col_idx]
    # Build table_data
    if headers and all(h.strip() for h in headers):
        table_data = [dict(zip(headers, row)) for row in data_rows if len(row) == len(headers)]
    else:
        table_data = data_rows
        headers = None
    return table_data, headers

def synthesize_cross_chunk_table(table_chunks):
    """
    Combine table data from multiple consecutive table chunks with similar headers/section.
    Returns (combined_table_data, unified_headers, section, page_range)
    """
    if not table_chunks:
        return None, None, None, None
    # Use the most common headers among chunks
    header_counts = {}
    for chunk in table_chunks:
        h = tuple(chunk.get('table_headers') or [])
        header_counts[h] = header_counts.get(h, 0) + 1
    unified_headers = max(header_counts, key=header_counts.get) if header_counts else None
    # Combine all rows with matching headers
    combined = []
    for chunk in table_chunks:
        if tuple(chunk.get('table_headers') or []) == unified_headers and chunk.get('table_data'):
            combined.extend(chunk['table_data'])
    # Deduplicate rows
    seen = set()
    deduped = []
    for row in combined:
        row_tuple = tuple(row.items()) if isinstance(row, dict) else tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            deduped.append(row)
    # Section and page range
    section = table_chunks[0].get('section_title') or table_chunks[0].get('chapter_title') or 'Unknown Section'
    pages = [c.get('page_number') for c in table_chunks if c.get('page_number') is not None]
    if pages:
        page_range = f"p.{min(pages)+1}-{max(pages)+1}" if len(set(pages)) > 1 else f"p.{pages[0]+1}"
    else:
        page_range = "?"
    return deduped, list(unified_headers) if unified_headers else None, section, page_range

def chunk_document(text: str, doc_id: str, filename: str, chunk_size: int = 500, chunk_overlap: int = 50, structured_pages: list = None, toc_data: list = None) -> List[Dict]:
    """
    Splits text into context-aware chunks with comprehensive metadata.
    Accepts optional structured_pages for chapter/section metadata.
    Accepts toc_data for robust chapter/section tagging.
    """
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    text = normalize_text(text)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    text_chunks = text_splitter.split_text(text)
    page_offsets = []
    if structured_pages:
        offset = 0
        for page in structured_pages:
            page_len = len(normalize_text(page['page_text']))
            page_offsets.append((offset, offset + page_len, page))
            offset += page_len
    toc_page_map = {}
    if toc_data:
        sorted_toc = sorted(toc_data, key=lambda x: x['page'])
        for idx, entry in enumerate(sorted_toc):
            start_page = entry['page']
            end_page = sorted_toc[idx + 1]['page'] if idx + 1 < len(sorted_toc) else 9999
            for p in range(start_page, end_page):
                toc_page_map[p] = entry
    heading_map = {}
    if not toc_data and structured_pages:
        heading_pattern = re.compile(r'^\s*(\d+(?:\.\d+)*)(?:\s+)([A-Za-z].{3,})', re.MULTILINE)
        for page in structured_pages:
            for match in heading_pattern.finditer(page['page_text']):
                number, title = match.groups()
                heading_map[page['page_num']] = {'number': number, 'title': title}
    chunks = []
    running_offset = 0
    for i, chunk_content in enumerate(text_chunks):
        chunk_id = hashlib.sha256(f"{doc_id}-{i}-{chunk_content}".encode('utf-8')).hexdigest()
        chapter_title = None
        chapter_number = None
        section_title = None
        page_number = None
        section_heading = None
        is_table_chunk = False
        table_data = None
        table_headers = None
        # Assign metadata using structured_pages
        if structured_pages:
            chunk_start = running_offset
            chunk_end = running_offset + len(chunk_content)
            for start, end, page in page_offsets:
                if start <= chunk_start < end:
                    page_number = page.get('page_num')
                    section_title = page.get('section_title')
                    break
        # Use TOC to assign chapter/section metadata
        if toc_data and page_number is not None:
            # Find the closest TOC entry whose page is <= page_number+1
            toc_entry = None
            for entry in reversed(sorted(toc_data, key=lambda x: x['page'])):
                if entry['page'] <= (page_number + 1):
                    toc_entry = entry
                    break
            if toc_entry:
                chapter_number = toc_entry['number']
                chapter_title = toc_entry['title']
        # Fallback: use heading_map if no TOC
        if not toc_data and page_number is not None and page_number in heading_map:
            chapter_number = heading_map[page_number]['number']
            chapter_title = heading_map[page_number]['title']
        # Extract sub-section heading from chunk content (e.g., 2.1 Some Subsection)
        subsection_match = re.match(r'^\s*(\d+\.\d+)\s+(.+)', chunk_content.strip())
        if subsection_match:
            section_heading = subsection_match.group(0).strip()
        # Try to match section_heading from TOC if not found
        if toc_data and not section_heading and page_number is not None:
            for entry in toc_data:
                if entry['page'] == (page_number + 1) and re.match(r'^\d+\.\d+', entry['number']):
                    section_heading = f"{entry['number']} {entry['title']}"
                    break
        # Heuristic for tabular data: high density of short lines, delimiters, or columns
        lines = chunk_content.splitlines()
        short_lines = [l for l in lines if len(l.strip()) > 0 and len(l.strip()) < 40]
        delimiter_lines = [l for l in lines if '\t' in l or '|' in l or ',' in l]
        multi_column_lines = [l for l in lines if len(l.split()) > 4]
        if (len(short_lines) > 3 and len(short_lines) / max(1, len(lines)) > 0.5) or len(delimiter_lines) > 2 or len(multi_column_lines) > 2:
            is_table_chunk = True
            table_data, table_headers = parse_table_from_text(chunk_content)
        chunk_dict = {
            'content': chunk_content,
            'doc_id': doc_id,
            'source': filename,
            'chunk_id': chunk_id,
            'chunk_index': i,
            'chunk_size': len(chunk_content),
            'page_number': page_number,
            'chapter_title': chapter_title,
            'chapter_number': chapter_number,
            'section_title': section_title,
            'section_heading': section_heading,
            'is_table_chunk': is_table_chunk,
            'table_data': table_data,
            'table_headers': table_headers,
            'timestamp_hint': None
        }
        chunks.append(chunk_dict)
        running_offset += len(chunk_content)
    return chunks

# =============================================================================
# EMBEDDING GENERATION AND STORAGE FUNCTIONS
# =============================================================================

def generate_embeddings(chunks: List[Dict]) -> List[Dict]:
    """
    Generates vector embeddings for a list of chunk dictionaries.
    
    This function uses the SentenceTransformer model to convert text chunks into
    high-dimensional vector representations that capture semantic meaning. The
    embeddings are added to each chunk dictionary for storage and retrieval.
    
    Args:
        chunks (List[Dict]): List of chunk dictionaries containing 'content' field
        
    Returns:
        List[Dict]: List of chunk dictionaries with embeddings added
        
    Note:
        - Requires internet connection for first-time model download
        - Embeddings are 384-dimensional vectors (all-MiniLM-L6-v2 model)
        - Processed in batches for efficiency
        
    Example:
        >>> chunks_with_embeddings = generate_embeddings(text_chunks)
        >>> print(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
    """
    # Validate embedding model availability
    if embedding_model is None:
        print("Error: Embedding model not loaded")
        return chunks
    
    if not chunks:
        print("Warning: No chunks provided for embedding generation")
        return chunks
    
    try:
        # Extract content for embedding generation
        contents = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings using SentenceTransformer
        print(f"Generating embeddings for {len(contents)} chunks...")
        embeddings = embedding_model.encode(contents, show_progress_bar=True)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
        
        print(f"✅ Generated embeddings for {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return chunks

def save_embeddings(embeddings: List[Dict], storage_name: str) -> Tuple[bool, Optional[str]]:
    """
    Persistently saves a list of embedding-chunk dictionaries to disk.
    
    This function ensures the embeddings directory exists and saves the data
    using pickle format for efficient storage and retrieval. It provides
    comprehensive error handling for file operations.
    
    Args:
        embeddings (List[Dict]): List of chunk dictionaries with embeddings
        storage_name (str): Name for the storage file (without extension)
        
    Returns:
        Tuple[bool, Optional[str]]: 
            - First element: True if successful, False otherwise
            - Second element: Error message if failed, None if successful
            
    Note:
        - Files are saved in the embeddings/ directory
        - Storage name should be alphanumeric and safe for filenames
        - Data is serialized using pickle for efficiency
        
    Example:
        >>> success, error = save_embeddings(chunks, "my_documents")
        >>> if success:
        ...     print("Embeddings saved successfully")
        ... else:
        ...     print(f"Error: {error}")
    """
    try:
        # Validate input parameters
        if not embeddings:
            return False, "No embeddings provided for saving"
        
        if not storage_name or not storage_name.strip():
            return False, "Storage name cannot be empty"
        
        # Sanitize storage name for filename safety
        safe_name = "".join(c for c in storage_name if c.isalnum() or c in ('-', '_')).rstrip()
        if not safe_name:
            return False, "Storage name contains no valid characters"
        
        # Ensure embeddings directory exists
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        
        # Create file path
        file_path = os.path.join(EMBEDDINGS_DIR, f"{safe_name}.pkl")
        
        # Save to pickle file with error handling
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"✅ Embeddings saved to {file_path}")
        return True, None
        
    except PermissionError:
        error_msg = f"Permission denied: Cannot write to {EMBEDDINGS_DIR}"
        print(f"❌ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Error saving embeddings: {e}"
        print(f"❌ {error_msg}")
        return False, error_msg

def load_embeddings(storage_name: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Loads a list of embedding-chunk dictionaries from a pickle file.
    
    This function provides comprehensive error handling for file operations
    and validates the loaded data structure. It supports both relative and
    absolute file paths.
    
    Args:
        storage_name (str): Name of the storage file (without .pkl extension)
        
    Returns:
        Tuple[Optional[List[Dict]], Optional[str]]: 
            - First element: Loaded embeddings data if successful, None if failed
            - Second element: Error message if failed, None if successful
            
    Note:
        - Files are expected to be in the embeddings/ directory
        - Data is deserialized using pickle
        - Validates data structure after loading
        
    Example:
        >>> data, error = load_embeddings("my_documents")
        >>> if error:
        ...     print(f"Error loading: {error}")
        ... else:
        ...     print(f"Loaded {len(data)} chunks")
    """
    try:
        # Validate input parameter
        if not storage_name or not storage_name.strip():
            return None, "Storage name cannot be empty"
        
        # Sanitize storage name
        safe_name = "".join(c for c in storage_name if c.isalnum() or c in ('-', '_')).rstrip()
        if not safe_name:
            return None, "Storage name contains no valid characters"
        
        # Create file path
        file_path = os.path.join(EMBEDDINGS_DIR, f"{safe_name}.pkl")
        
        # Check if file exists
        if not os.path.exists(file_path):
            return None, f"Embedding file not found: {file_path}"
        
        # Load data from pickle file
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        # Validate loaded data structure
        if not isinstance(embeddings, list):
            return None, "Invalid data format: expected list of dictionaries"
        
        if embeddings and not isinstance(embeddings[0], dict):
            return None, "Invalid data format: expected list of dictionaries"
        
        print(f"✅ Embeddings loaded from {file_path}")
        return embeddings, None
        
    except PermissionError:
        error_msg = f"Permission denied: Cannot read from {file_path}"
        print(f"❌ {error_msg}")
        return None, error_msg
    except pickle.UnpicklingError as e:
        error_msg = f"Error unpickling file: {e}"
        print(f"❌ {error_msg}")
        return None, error_msg
    except Exception as e:
        error_msg = f"Error loading embeddings: {e}"
        print(f"❌ {error_msg}")
        return None, error_msg

def get_available_embedding_sets() -> List[str]:
    """
    Returns a list of available embedding set names from the embeddings directory.
    
    This function scans the embeddings directory for .pkl files and returns
    their names without the file extension. It provides error handling for
    directory access issues.
    
    Returns:
        List[str]: List of available embedding set names (without .pkl extension)
        
    Note:
        - Only returns .pkl files from the embeddings/ directory
        - File extensions are automatically removed
        - Returns empty list if directory doesn't exist or is empty
        
    Example:
        >>> available_sets = get_available_embedding_sets()
        >>> print(f"Available sets: {available_sets}")
    """
    try:
        # Check if embeddings directory exists
        if not os.path.exists(EMBEDDINGS_DIR):
            return []
        
        # List all files in the directory
        files = os.listdir(EMBEDDINGS_DIR)
        
        # Filter for .pkl files and remove extensions
        embedding_files = [f for f in files if f.endswith('.pkl')]
        return [f[:-4] for f in embedding_files]  # Remove .pkl extension
        
    except PermissionError:
        print(f"Permission denied: Cannot access {EMBEDDINGS_DIR}")
        return []
    except Exception as e:
        print(f"Error listing embedding sets: {e}")
        return []

# =============================================================================
# VECTOR STORE CLASS
# =============================================================================

class InMemoryVectorStore:
    """
    A simple in-memory vector store for efficient similarity search.
    
    This class provides an in-memory implementation of a vector store that
    supports adding embeddings and performing cosine similarity search.
    It's designed for prototyping and small to medium-sized datasets.
    
    Attributes:
        chunks_data (List[Dict]): List of chunk dictionaries with metadata
        embeddings_matrix (np.ndarray): 2D array of embeddings for similarity search
        
    Note:
        - Uses cosine similarity for semantic search
        - Stores embeddings as NumPy arrays for efficient computation
        - Supports incremental addition of new embeddings
        - Memory usage scales with number of embeddings and vector dimension
    """
    
    def __init__(self):
        """
        Initialize an empty vector store.
        
        Creates empty containers for chunks data and embeddings matrix.
        The embeddings matrix will be initialized when first vectors are added.
        """
        self.chunks_data = []  # List of dictionaries containing chunk metadata
        self.embeddings_matrix = np.array([])  # 2D NumPy array of embeddings
    
    def add_vectors(self, embeddings_data: List[Dict]):
        """
        Adds a list of chunk dictionaries with embeddings to the vector store.
        
        This method filters out chunks without embeddings, extracts the embedding
        vectors, and appends them to the existing embeddings matrix. It maintains
        the relationship between chunks_data and embeddings_matrix.
        
        Args:
            embeddings_data (List[Dict]): List of chunk dictionaries with 'embedding' field
            
        Note:
            - Only chunks with valid embeddings are added
            - Embeddings are converted to NumPy arrays for efficient computation
            - Matrix is extended vertically to accommodate new embeddings
            - Maintains data integrity between chunks and embeddings
            
        Example:
            >>> vector_store = InMemoryVectorStore()
            >>> vector_store.add_vectors(chunks_with_embeddings)
            >>> print(f"Added {len(vector_store.chunks_data)} vectors")
        """
        # Filter out chunks missing embeddings
        valid_chunks = [chunk for chunk in embeddings_data if 'embedding' in chunk]
        
        if not valid_chunks:
            print("Warning: No valid chunks with embeddings found")
            return
        
        # Extract embeddings and convert to NumPy array
        new_embeddings = np.array([chunk['embedding'] for chunk in valid_chunks])
        
        # Append to chunks_data
        self.chunks_data.extend(valid_chunks)
        
        # Stack embeddings into embeddings_matrix
        if self.embeddings_matrix.size == 0:
            # First addition: initialize the matrix
            self.embeddings_matrix = new_embeddings
        else:
            # Subsequent additions: stack vertically
            self.embeddings_matrix = np.vstack([self.embeddings_matrix, new_embeddings])
        
        print(f"✅ Added {len(valid_chunks)} vectors to vector store")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Performs cosine similarity search to find the most similar chunks.
        
        This method computes cosine similarity between the query embedding and
        all stored embeddings, then returns the top-k most similar chunks.
        It handles edge cases like empty stores and invalid inputs.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            k (int): Number of top results to return (default: 5)
            
        Returns:
            List[Dict]: List of top-k most similar chunk dictionaries
            
        Note:
            - Uses sklearn's cosine_similarity for efficient computation
            - Returns empty list if no embeddings are stored
            - k is automatically adjusted if fewer embeddings are available
            - Results are sorted by similarity score (highest first)
            
        Example:
            >>> query_vec = embedding_model.encode(["What is AI?"])[0]
            >>> results = vector_store.search(query_vec, k=3)
            >>> print(f"Found {len(results)} similar chunks")
        """
        # Check if vector store has embeddings
        if self.embeddings_matrix.size == 0:
            print("Warning: No embeddings in vector store")
            return []
        
        # Validate input parameters
        if k <= 0:
            print("Warning: k must be positive, using k=1")
            k = 1
        
        # Adjust k if fewer embeddings are available
        k = min(k, len(self.chunks_data))
        
        try:
            # Reshape query embedding if needed (ensure 2D array)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Calculate cosine similarities using sklearn
            similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
            
            # Get top k indices (highest similarity first)
            top_indices = np.argsort(similarities)[::-1][:k]
            
            # Return top k chunks
            results = [self.chunks_data[i] for i in top_indices]
            
            return results
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

# =============================================================================
# GEMINI INTEGRATION FUNCTION
# =============================================================================

def rerank_chunks(query: str, chunks_to_rerank: list, top_k_reranked: int = 7) -> list:
    """
    Reranker: Uses a cross-encoder to score and select the most relevant chunks from a candidate pool.
    Args:
        query (str): The user query string.
        chunks_to_rerank (list): Candidate chunk dicts to rerank.
        top_k_reranked (int): Number of top reranked results to return.
    Returns:
        list: Top reranked chunk dicts, sorted by cross-encoder relevance.
    """
    if not chunks_to_rerank or not query or reranker_model is None:
        return chunks_to_rerank[:top_k_reranked]
    pairs = [(query, chunk.get('content', '')) for chunk in chunks_to_rerank]
    scores = reranker_model.predict(pairs)
    scored_chunks = list(zip(scores, chunks_to_rerank))
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored_chunks[:top_k_reranked]]

# Enhanced Gemini answer generation with reranking, self-critique, and structured markdown

def headers_are_similar(headers1, headers2, threshold=0.7):
    """
    Return True if two header lists are similar enough (using difflib SequenceMatcher ratio).
    """
    if not headers1 or not headers2:
        return False
    if abs(len(headers1) - len(headers2)) > 1:
        return False
    matches = 0
    for h1 in headers1:
        best = max((difflib.SequenceMatcher(None, h1.lower(), h2.lower()).ratio() for h2 in headers2), default=0)
        if best >= threshold:
            matches += 1
    return matches >= min(len(headers1), len(headers2)) * threshold

def is_table_continuation(chunk):
    """
    Returns True if the chunk's headers or first line indicate a table continuation.
    """
    headers = chunk.get('table_headers') or []
    if any(re.search(r'(continued|cont\.?|continue)', h, re.IGNORECASE) for h in headers):
        return True
    content = chunk.get('content', '').splitlines()
    if content and re.search(r'(continued|cont\.?|continue)', content[0], re.IGNORECASE):
        return True
    return False

def stitch_horizontal_tables(table_chunks, row_tolerance=1, header_threshold=0.7):
    """
    Horizontally stitch consecutive table chunks on the same page/section with similar row counts and fuzzy-matched headers.
    Also merges chunks marked as table continuations.
    Returns a list of stitched table chunks (each with merged headers and rows).
    """
    if not table_chunks:
        return []
    stitched = []
    used = set()
    for i, chunk in enumerate(table_chunks):
        if i in used:
            continue
        group = [chunk]
        base_rows = chunk.get('table_data')
        base_headers = chunk.get('table_headers') or []
        base_page = chunk.get('page_number')
        base_section = chunk.get('section_title') or chunk.get('chapter_title')
        for j in range(i+1, len(table_chunks)):
            other = table_chunks[j]
            if j in used:
                continue
            # Same page/section
            same_page = other.get('page_number') == base_page
            same_section = (other.get('section_title') or other.get('chapter_title')) == base_section
            # Similar row count
            other_rows = other.get('table_data')
            other_headers = other.get('table_headers') or []
            if base_rows and other_rows and same_page and same_section:
                if abs(len(base_rows) - len(other_rows)) <= row_tolerance:
                    if headers_are_similar(base_headers, other_headers, threshold=header_threshold) or is_table_continuation(other):
                        group.append(other)
                        used.add(j)
        if len(group) == 1:
            stitched.append(chunk)
        else:
            # Merge headers and rows horizontally
            all_headers = []
            all_rows = []
            for g in group:
                all_headers.extend(g.get('table_headers') or [])
            # Pad rows to max length
            max_rows = max(len(g.get('table_data') or []) for g in group)
            group_rows = [g.get('table_data') or [] for g in group]
            for row_idx in range(max_rows):
                stitched_row = []
                for rows in group_rows:
                    if row_idx < len(rows):
                        row = rows[row_idx]
                        if isinstance(row, dict):
                            stitched_row.extend(list(row.values()))
                        else:
                            stitched_row.extend(row)
                    else:
                        stitched_row.extend([''] * (len(rows[0]) if rows and isinstance(rows[0], list) else 1))
                all_rows.append(stitched_row)
            stitched.append({
                'content': '\n'.join([str(r) for r in all_rows]),
                'table_headers': all_headers,
                'table_data': all_rows,
                'is_table_chunk': True,
                'page_number': base_page,
                'section_title': base_section,
                'chapter_title': chunk.get('chapter_title'),
                'chapter_number': chunk.get('chapter_number'),
            })
        used.add(i)
    return stitched

def table_confidence_score(chunk):
    """
    Assign a confidence score (0-1) to a table chunk based on header consistency, row/column regularity, and delimiter clarity.
    Higher is better.
    """
    headers = chunk.get('table_headers') or []
    data = chunk.get('table_data') or []
    if not headers or not data:
        return 0.0
    # Header consistency: all headers are non-empty and unique
    header_score = 1.0 if all(h.strip() for h in headers) and len(set(headers)) == len(headers) else 0.5
    # Row/column regularity: most rows have the same length as headers
    row_lengths = [len(row) if isinstance(row, list) else len(row.values()) for row in data]
    reg_score = sum(1 for l in row_lengths if l == len(headers)) / max(1, len(row_lengths))
    # Delimiter clarity: if original content has clear delimiters
    content = chunk.get('content', '')
    delim_score = 1.0 if ('|' in content or '\t' in content or ',' in content) else 0.5
    # Final score: weighted average
    return 0.4 * header_score + 0.4 * reg_score + 0.2 * delim_score

def filter_table_data(table_data, headers, query):
    """
    Filter table rows based on query keywords (e.g., 'type string' -> only rows where any cell contains 'string').
    Returns filtered table_data and a filter description.
    """
    import re
    qwords = [w for w in re.split(r'\W+', query.lower()) if w]
    if not qwords:
        return table_data, None
    filtered = []
    for row in table_data:
        row_values = [str(row.get(h, '') if isinstance(row, dict) else cell).lower() for h, cell in zip(headers, row)]
        if any(any(q in v for v in row_values) for q in qwords):
            filtered.append(row)
    if filtered and len(filtered) < len(table_data):
        return filtered, f"Filtered rows by query keywords: {', '.join(qwords)}"
    return table_data, None

def summarize_table(table_data, headers, n=5):
    """
    Return the top n rows and, if possible, the most common values per column.
    """
    summary = table_data[:n]
    col_summaries = {}
    if headers and table_data:
        for idx, h in enumerate(headers):
            col = [str(row.get(h, '') if isinstance(row, dict) else row[idx]) for row in table_data if (isinstance(row, dict) and h in row) or (isinstance(row, list) and idx < len(row))]
            most_common = Counter(col).most_common(1)
            if most_common:
                col_summaries[h] = most_common[0][0]
    return summary, col_summaries

def highlight_table_cells(table_data, headers, query):
    """
    Bold cells in the table that match any query keyword.
    Returns a new table_data with markdown bolding applied.
    """
    import re
    qwords = [w for w in re.split(r'\W+', query.lower()) if w]
    if not qwords:
        return table_data
    highlighted = []
    for row in table_data:
        new_row = []
        for idx, cell in enumerate(row if isinstance(row, list) else [row.get(h, '') for h in headers]):
            cell_str = str(cell)
            if any(q in cell_str.lower() for q in qwords):
                cell_str = f"**{cell_str}**"
            new_row.append(cell_str)
        highlighted.append(new_row)
    return highlighted

def is_table_query(query: str) -> bool:
    """
    Determine if a query is asking for tabular data.
    
    Args:
        query (str): The user query
        
    Returns:
        bool: True if the query is asking for tabular data
    """
    table_query_keywords = ["table", "list", "fields", "columns", "parameters", "matrix", "mapping"]
    return any(kw in query.lower() for kw in table_query_keywords)

def is_procedural_query(query: str) -> bool:
    """
    Determine if a query is asking for procedural steps.
    
    Args:
        query (str): The user query
        
    Returns:
        bool: True if the query is asking for procedural steps
    """
    procedural_keywords = ["how to", "steps to", "configure", "setup", "create", "procedure", "process", "set up", "step by step", "instruction", "install", "enable", "register", "initialize", "make", "add", "remove", "delete", "update", "edit"]
    return any(kw in query.lower() for kw in procedural_keywords)

def generate_answer_with_gemini(query: str, context_chunks: list, query_embedding: np.ndarray, all_chunks: list, rerank_top_k: int = 7) -> str:
    """
    Generate a hyper-direct, context-bound answer with Gemini, with special handling for tabular and procedural queries.
    - Extreme directness: No filler, start with answer.
    - Strict context: Use ONLY 'Document Context'. State clearly if info is not found.
    - Tabular: Present as markdown table or key-value pairs, not description.
    - Citation: Cite page number(s) after each fact.
    - Procedural: List steps as numbered points, exactly as in doc.
    """
    is_table_query_result = is_table_query(query)
    is_procedural_query_result = is_procedural_query(query)
    # 1. Rerank top-K chunks
    reranked_chunks = rerank_chunks(query, context_chunks, top_k_reranked=rerank_top_k)
    # 2. If no high-confidence match, trigger a broader re-retrieval
    high_confidence = any(cosine_similarity([query_embedding], [np.array(chunk['embedding'])])[0][0] > 0.55 for chunk in reranked_chunks)
    if not high_confidence:
        reranked_chunks = rerank_chunks(query, all_chunks, top_k_reranked=rerank_top_k)
    # 3. Table-aware answer synthesis (cross-chunk, horizontal stitching)
    table_chunks = [c for c in reranked_chunks if c.get('is_table_chunk') and c.get('table_data')]
    if is_table_query_result and table_chunks:
        stitched_tables = stitch_horizontal_tables(table_chunks)
        best_table = max(stitched_tables, key=lambda t: table_confidence_score(t), default=None)
        confidence = table_confidence_score(best_table) if best_table else 0.0
        if best_table and best_table.get('table_headers') and best_table.get('table_data'):
            headers = best_table['table_headers']
            table_data = best_table['table_data']
            section = best_table.get('section_title') or best_table.get('chapter_title') or 'Unknown Section'
            page = best_table.get('page_number')
            page_str = f"p.{page+1}" if page is not None else "?"
            # Highlighting
            table_data = highlight_table_cells(table_data, headers, query)
            md = f"| " + " | ".join(headers) + " |\n"
            md += "|" + "|".join(["---"] * len(headers)) + "|\n"
            for row in table_data:
                md += "| " + " | ".join(str(cell) for cell in row) + " |\n"
            md += f"> (from section {section}, {page_str})\n\n"
            if confidence < 0.7:
                md += f"⚠️ Table structure inferred; verify for accuracy. (Confidence: {confidence:.2f})\n"
            md += "[Strictly extracted from Document Context. No additional commentary.]\n"
            return md
    # 4. Build hyper-specific prompt for Gemini
    context_md = ""
    for i, chunk in enumerate(reranked_chunks, 1):
        section = chunk.get('section_title') or chunk.get('chapter_title') or 'Unknown Section'
        page = chunk.get('page_number')
        page_str = f"p.{page+1}" if page is not None else "?"
        context_md += f"### 🔹 {section}\n- {chunk['content']}\n> (from section {section}, {page_str})\n\n"
    # --- NEW PROMPT (Persona, Tone, and Structure Directives) ---
#     prompt = f"""
# You are a helpful, professional, and clear technical expert, dedicated to guiding the user through complex documentation. Your primary goal is to answer the 'User Query' directly and concisely, drawing *only* from the 'Document Context' provided, answer should be human like and not robotic.

# Strict Instructions for a Human-like and Accurate Response:
# 1.  **Direct & Professional Answer:** Begin your response directly with the answer to the question. Use clear, concise, and professional language. Avoid overly robotic or overly casual phrasing.
# 2.  **Context Bound & Cited:** Refer *solely* to the 'Document Context'. Do NOT use outside knowledge or infer beyond the given text. For *every piece of information* provided, cite the original page number(s) from the document immediately after the relevant phrase or sentence.
# 3.  **Tabular Data Clarity:** If the 'Document Context' contains tables relevant to the query (indicated by is_table_chunk=True in chunk metadata or clear formatting), extract the specific data points requested or relevant rows/columns and present them clearly. You MUST use Markdown table format if appropriate, or a concise list of key-value pairs if not. Do not just describe the table's existence; extract its content.
# 4.  **Procedural Steps:** For procedural instructions or 'how-to' questions, list the steps clearly using numbered lists or bullet points, following the original document's flow as closely as possible.
# 5.  **Nuance & Confidence:** If the information's certainty is not absolute based on the context, use cautious phrasing like 'It appears that...', 'The document suggests...', or 'Based on the provided information, it seems...'. If the document does not contain the answer, state that clearly and concisely, e.g., "The document does not provide information on...".
# 6.  **Vary Phrasing:** Vary sentence structure and vocabulary to enhance readability and natural flow. Avoid repetitive sentence beginnings.

# Document Context (Note: Each chunk might have associated metadata like page numbers):
# {context_md}

# User Query:
# {query}

# Answer (professional, clear, and directly from context, with citations):
# """
    prompt = f"""
You are a friendly, highly knowledgeable, and exceptionally helpful virtual assistant specializing in technical documentation. Your goal is to guide the user effectively and make complex information approachable. Your paramount objective is always factual accuracy, grounding in the provided 'Document Context', and precise citation. Never hallucinate or infer information not explicitly present.

Strict Instructions for a Human-like, Conversational, and Accurate Response:

I. Core Persona & Foundational Principles:

1.  **Warm Acknowledgment:** Begin with a friendly, varied greeting that acknowledges the user's question (e.g., "Happy to help with that!", "Great question about...", "I'd be glad to explain..."). Avoid repetitive openings across conversations.

2.  **Direct & Conversational Answer:** Provide clear, concise information in a warm, approachable tone. Use natural language patterns with occasional contractions (I'll, that's, here's) and thoughtful transitions between ideas. Balance professionalism with conversational flow.

3.  **Context-Bound & Cited:** Draw information EXCLUSIVELY from the 'Document Context'. For EVERY fact or piece of information, include the page number citation immediately after the relevant sentence (e.g., "The system supports PDF and TXT documents (p.3)."). Never fabricate information not present in the context.

II. Natural Language Flow:

4.  **Contractions:** Use common contractions (I'm, you're, it's, we'll) to make language feel more conversational and less formal.

5.  **Sentence Variety:** Mix short, direct sentences with longer, more descriptive ones. Vary sentence beginnings to avoid monotony.

6.  **Rhythm and Pacing:** Use natural transitions ("however," "therefore," "in addition," "moving on to") to create logical flow. Structure information with clear topic sentences and appropriate pacing.

III. Tone and Voice:

7.  **Consistent Voice:** Maintain a consistently friendly, professional, and empathetic tone throughout. Be encouraging and patient.

8.  **Emotionally Aware Language:** If the user expresses frustration or asks about a complex topic, acknowledge their situation empathetically (e.g., "I understand this can be complex," "I'm here to help you navigate this").

IV. Context Awareness:

9.  **Relevance to Previous Conversation:** If the query builds on previous discussion (when context is provided), acknowledge this connection (e.g., "Building on our previous discussion about...").

10. **Memory of Prior Facts:** Reference key facts from earlier in the conversation if relevant and included in the context.

11. **Situational Appropriateness:** Adapt detail level based on the implied complexity of the user's query and document depth.

V. Clarity and Structure:

12. **Clear Sentence Structure:** Use straightforward grammar and syntax. Avoid overly complex clauses where simpler phrasing would suffice.

13. **Technical Terms:** Explain specialized terminology when first introduced. If the document uses acronyms (like "FSM"), ensure you clarify their meaning within the document's context.

14. **Structured Clarity:** 
   - For tabular data: Present in clean markdown tables with clear headers
   - For procedures: Use numbered steps with concise descriptions
   - For conceptual information: Use short paragraphs with occasional bullet points for key concepts

VI. Empathy and Engagement:

15. **Acknowledgment of Complexity:** For complex topics, briefly acknowledge the complexity before explaining clearly (e.g., "This is a nuanced topic, but I'll break it down...").

16. **Validation and Support:** Offer subtle validation for questions (e.g., "That's a good question," "Many users find this part challenging").

17. **Rhetorical Questions:** Occasionally use well-placed rhetorical questions to guide the user or make the response more dynamic, but only when it genuinely adds clarity.

18. **Addressing Directly:** Use "you" and "your" naturally to address the user directly, fostering a sense of one-on-one interaction.

VII. Confidence and Conclusion:

19. **Nuanced Confidence:** When information is limited or uncertain based on the context, use thoughtful phrasing like "Based on the documentation, it appears that..." or "The document suggests...". If information is absent, clearly state: "I don't see specific information about that in the documentation."

20. **Helpful Conclusion:** End with a warm, helpful closing that invites further questions while being concise (e.g., "Hope that helps! Let me know if you need any clarification.", "Anything else you'd like to know about this topic?").

VIII. Important Caveats:

21. **Avoid Idioms and Slang:** Strictly avoid idioms, slang, colloquialisms, or specific cultural references that could be misunderstood or sound unprofessional.

22. **Real-world Logic:** Ensure responses reflect practical applicability. If a step seems logically dependent on another not explicitly stated, mention the dependency.

23. **Avoid Mechanical Outputs:** Integrate information smoothly rather than simply listing facts in a detached manner.

Document Context:
{context_md}

User Query:
{query}

Answer:
"""
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    response = model.generate_content(prompt)
    if response and response.text:
        return response.text.strip()
    else:
        return "Error: Empty response from Gemini API"

# For backward compatibility

generate_structured_answer_with_gemini = generate_answer_with_gemini
def refine_answer_with_gemini(answer: str) -> str:
    """
    Refines a given answer using a second LLM call to make it more eloquent,
    insightful, and authoritative, with a natural tone.

    Args:
        answer (str): The initial answer generated by the RAG pipeline.

    Returns:
        str: The refined, "super-human" response.
    """
    if not answer or not isinstance(answer, str):
        return answer

    refinement_prompt = f"""
    You are a friendly, knowledgeable assistant who explains things in a warm, conversational way. Transform the following response to sound more human, approachable, and engaging while keeping all the important information.

    **Original Response:**
    {answer}

    **Make it more human by:**
    1. **Conversational Tone:** Use "you" and "I" naturally, like talking to a friend who asked for help
    2. **Warm & Friendly:** Add encouraging phrases, show empathy, and be supportive in your explanations  
    3. **Natural Language:** Use everyday words and phrases people actually use in conversation
    4. **Personal Touch:** Add transitional phrases like "Here's what I found...", "Let me explain...", "What's interesting is..."
    5. **Helpful Attitude:** Show genuine interest in helping and make the person feel comfortable asking questions

    **Guidelines:**
    - Keep all factual information and citations intact
    - Make it sound like a helpful human expert, not an AI
    - Use contractions (don't, can't, you'll) to sound more natural
    - Add gentle explanations for complex terms
    - Show enthusiasm for the topic when appropriate

    **Human-Friendly Response:**
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(refinement_prompt)
        if response and response.text:
            return response.text.strip()
        return answer  # Fallback to original answer
    except Exception as e:
        print(f"Error during answer refinement: {e}")
        return answer  # Fallback to original answer

def estimate_tokens(text: str) -> int:
    # Try tiktoken, fallback to char/4 (safe for most LLMs)
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) / 4))

def iterative_summarize_chunks(chunks, query, llm_model=None, max_tokens_per_group=6000, reduce_batch_size=5):
    if not chunks:
        return "No content to summarize."
    def group_chunks_by_tokens(chunks, max_tokens):
        groups, current, total = [], [], 0
        for chunk in chunks:
            chunk_tokens = estimate_tokens(chunk['content'])
            if total + chunk_tokens > max_tokens and current:
                groups.append(current)
                current, total = [], 0
            current.append(chunk)
            total += chunk_tokens
        if current:
            groups.append(current)
        return groups
    groups = group_chunks_by_tokens(chunks, max_tokens_per_group)
    improved_summary_prompt = (
        f"{query}\n\nSummarize the following content in a clear, multi-paragraph format. "
        "Use bullet points for key facts, and include a brief conclusion. "
        "If the content is technical, explain terms simply."
    )
    if len(groups) == 1:
        return generate_answer_with_gemini(improved_summary_prompt, groups[0], None, None)
    else:
        partial_summaries = []
        for group in groups:
            summary = generate_answer_with_gemini(improved_summary_prompt, group, None, None)
            partial_summaries.append({'content': summary})
        # Efficient reduce: group partial summaries into larger batches
        reduced_groups = [partial_summaries[i:i+reduce_batch_size] for i in range(0, len(partial_summaries), reduce_batch_size)]
        reduced_chunks = [{'content': ' '.join([s['content'] for s in group])} for group in reduced_groups]
        return iterative_summarize_chunks(reduced_chunks, query, llm_model, max_tokens_per_group, reduce_batch_size)

# Unified grouping and summarization helper

def group_and_summarize(chunks, group_key, summary_prompt):
    from collections import defaultdict
    group_map = defaultdict(list)
    for chunk in chunks:
        key = chunk.get(group_key) or f'Unknown {group_key.title()}'
        group_map[key].append(chunk)
    summaries = {}
    for key, group_chunks in group_map.items():
        summary = iterative_summarize_chunks(group_chunks, summary_prompt)
        # Store both summary and chunk metadata
        summaries[key] = {
            'summary': summary,
            'chunks': [
                {
                    'page_number': c.get('page_number'),
                    'start': c['content'][:60] + ('...' if len(c['content']) > 60 else ''),
                    'chunk_index': c.get('chunk_index')
                }
                for c in group_chunks
            ]
        }
    return summaries

# =============================================================================
# METRICS AND EVALUATION FUNCTIONS
# =============================================================================

def calculate_retrieval_metrics(query: str, retrieved_chunks: List[Dict], query_embedding: np.ndarray) -> Dict[str, float]:
    """
    Calculate various retrieval metrics for the RAG system including accuracy measures.
    
    Args:
        query (str): The user's query
        retrieved_chunks (List[Dict]): List of retrieved chunks with embeddings
        query_embedding (np.ndarray): Embedding of the query
        
    Returns:
        Dict[str, float]: Dictionary containing various metrics including accuracy
    """
    if not retrieved_chunks or query_embedding is None:
        return {
            'avg_similarity': 0.0,
            'max_similarity': 0.0,
            'min_similarity': 0.0,
            'similarity_std': 0.0,
            'num_chunks_retrieved': 0,
            'avg_chunk_length': 0.0,
            'total_context_length': 0,
            'retrieval_accuracy': 0.0,
            'precision_at_k': 0.0,
            'semantic_coherence': 0.0
        }
    
    # Calculate similarity scores
    similarities = []
    chunk_lengths = []
    total_context_length = 0
    
    for chunk in retrieved_chunks:
        if 'embedding' in chunk:
            chunk_embedding = np.array(chunk['embedding'])
            similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            similarities.append(similarity)
        
        chunk_length = len(chunk.get('content', ''))
        chunk_lengths.append(chunk_length)
        total_context_length += chunk_length
    
    # Calculate accuracy metrics
    retrieval_accuracy = calculate_retrieval_accuracy(similarities)
    precision_at_k = calculate_precision_at_k(similarities, k=len(similarities))
    semantic_coherence = calculate_semantic_coherence(retrieved_chunks, query_embedding)
    
    # Calculate metrics
    metrics = {
        'avg_similarity': np.mean(similarities) if similarities else 0.0,
        'max_similarity': np.max(similarities) if similarities else 0.0,
        'min_similarity': np.min(similarities) if similarities else 0.0,
        'similarity_std': np.std(similarities) if similarities else 0.0,
        'num_chunks_retrieved': len(retrieved_chunks),
        'avg_chunk_length': np.mean(chunk_lengths) if chunk_lengths else 0.0,
        'total_context_length': total_context_length,
        'retrieval_accuracy': retrieval_accuracy,
        'precision_at_k': precision_at_k,
        'semantic_coherence': semantic_coherence
    }
    
    return metrics

def calculate_retrieval_accuracy(similarities: List[float], threshold: float = 0.5) -> float:
    """
    Calculate retrieval accuracy based on similarity threshold.
    
    Args:
        similarities (List[float]): List of similarity scores
        threshold (float): Minimum similarity threshold for relevant results
        
    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    if not similarities:
        return 0.0
    
    relevant_count = sum(1 for sim in similarities if sim >= threshold)
    accuracy = relevant_count / len(similarities)
    return accuracy

def calculate_precision_at_k(similarities: List[float], k: int, threshold: float = 0.5) -> float:
    """
    Calculate Precision@K metric for retrieval quality.
    
    Args:
        similarities (List[float]): List of similarity scores
        k (int): Number of top results to consider
        threshold (float): Minimum similarity threshold for relevance
        
    Returns:
        float: Precision@K score (0.0 to 1.0)
    """
    if not similarities or k <= 0:
        return 0.0
    
    # Take top k similarities
    top_k_similarities = similarities[:k]
    relevant_count = sum(1 for sim in top_k_similarities if sim >= threshold)
    precision = relevant_count / len(top_k_similarities)
    return precision

def calculate_semantic_coherence(retrieved_chunks: List[Dict], query_embedding: np.ndarray) -> float:
    """
    Calculate semantic coherence between retrieved chunks and query.
    
    Args:
        retrieved_chunks (List[Dict]): List of retrieved chunks with embeddings
        query_embedding (np.ndarray): Query embedding vector
        
    Returns:
        float: Semantic coherence score (0.0 to 1.0)
    """
    if not retrieved_chunks or query_embedding is None:
        return 0.0
    
    # Calculate inter-chunk similarities
    chunk_embeddings = []
    for chunk in retrieved_chunks:
        if 'embedding' in chunk:
            chunk_embeddings.append(np.array(chunk['embedding']))
    
    if len(chunk_embeddings) < 2:
        return 0.0
    
    # Calculate average similarity between chunks
    inter_similarities = []
    for i in range(len(chunk_embeddings)):
        for j in range(i + 1, len(chunk_embeddings)):
            sim = cosine_similarity([chunk_embeddings[i]], [chunk_embeddings[j]])[0][0]
            inter_similarities.append(sim)
    
    # Calculate query-chunk coherence
    query_similarities = []
    for embedding in chunk_embeddings:
        sim = cosine_similarity([query_embedding], [embedding])[0][0]
        query_similarities.append(sim)
    
    # Combine inter-chunk and query-chunk coherence
    inter_coherence = np.mean(inter_similarities) if inter_similarities else 0.0
    query_coherence = np.mean(query_similarities) if query_similarities else 0.0
    
    # Weighted average (query coherence is more important)
    semantic_coherence = 0.7 * query_coherence + 0.3 * inter_coherence
    return semantic_coherence

def calculate_response_metrics(response: str, query: str, retrieved_chunks: List[Dict] = None) -> Dict[str, any]:
    """
    Calculate metrics for the generated response including accuracy measures.
    
    Args:
        response (str): Generated response
        query (str): Original query
        retrieved_chunks (List[Dict]): Retrieved chunks used for response generation
        
    Returns:
        Dict[str, any]: Dictionary containing response metrics including accuracy
    """
    if not response or not query:
        return {
            'response_length': 0,
            'response_word_count': 0,
            'query_length': 0,
            'query_word_count': 0,
            'response_to_query_ratio': 0.0,
            'has_sources': False,
            'estimated_reading_time': 0.0,
            'response_accuracy': 0.0,
            'context_utilization': 0.0,
            'answer_completeness': 0.0
        }
    
    # Basic text metrics
    response_length = len(response)
    response_words = len(response.split())
    query_length = len(query)
    query_words = len(query.split())
    
    # Check if response contains source citations
    has_sources = any(keyword in response.lower() for keyword in ['source', 'file:', 'chunk', 'document'])
    
    # Estimated reading time (average 200 words per minute)
    reading_time = response_words / 200.0
    
    # Calculate accuracy metrics
    response_accuracy = calculate_response_accuracy(response, query)
    context_utilization = calculate_context_utilization(response, retrieved_chunks) if retrieved_chunks else 0.0
    answer_completeness = calculate_answer_completeness(response, query)
    
    metrics = {
        'response_length': response_length,
        'response_word_count': response_words,
        'query_length': query_length,
        'query_word_count': query_words,
        'response_to_query_ratio': response_length / query_length if query_length > 0 else 0.0,
        'has_sources': has_sources,
        'estimated_reading_time': reading_time,
        'response_accuracy': response_accuracy,
        'context_utilization': context_utilization,
        'answer_completeness': answer_completeness
    }
    
    return metrics

def calculate_response_accuracy(response: str, query: str) -> float:
    """
    Calculate response accuracy based on query-response alignment.
    
    Args:
        response (str): Generated response
        query (str): Original query
        
    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    if not response or not query:
        return 0.0
    
    # Convert to lowercase for comparison
    response_lower = response.lower()
    query_lower = query.lower()
    
    # Extract key terms from query (simple approach)
    query_words = set(word.strip('.,!?;:') for word in query_lower.split() if len(word) > 2)
    
    # Count how many query terms appear in response
    matched_terms = sum(1 for word in query_words if word in response_lower)
    
    # Calculate accuracy based on term coverage
    if len(query_words) == 0:
        return 0.0
    
    term_coverage = matched_terms / len(query_words)
    
    # Bonus for having sources and structured response
    structure_bonus = 0.0
    if any(keyword in response_lower for keyword in ['source', 'according to', 'based on']):
        structure_bonus += 0.1
    if len(response.split()) >= 20:  # Reasonable response length
        structure_bonus += 0.1
    
    # Penalty for error messages
    error_penalty = 0.0
    if any(keyword in response_lower for keyword in ['error', 'sorry', 'cannot', "don't have"]):
        error_penalty = 0.3
    
    accuracy = min(1.0, term_coverage + structure_bonus - error_penalty)
    return max(0.0, accuracy)

def calculate_context_utilization(response: str, retrieved_chunks: List[Dict]) -> float:
    """
    Calculate how well the response utilizes the retrieved context.
    
    Args:
        response (str): Generated response
        retrieved_chunks (List[Dict]): Retrieved chunks
        
    Returns:
        float: Context utilization score (0.0 to 1.0)
    """
    if not response or not retrieved_chunks:
        return 0.0
    
    response_lower = response.lower()
    total_chunks = len(retrieved_chunks)
    utilized_chunks = 0
    
    # Check how many chunks have content referenced in the response
    for chunk in retrieved_chunks:
        chunk_content = chunk.get('content', '').lower()
        if not chunk_content:
            continue
        
        # Extract key phrases from chunk (simple approach)
        chunk_words = set(word.strip('.,!?;:') for word in chunk_content.split() if len(word) > 3)
        
        # Check if any significant words from chunk appear in response
        matches = sum(1 for word in chunk_words if word in response_lower)
        
        # If enough matches, consider chunk utilized
        if matches >= min(3, len(chunk_words) * 0.1):
            utilized_chunks += 1
    
    utilization = utilized_chunks / total_chunks if total_chunks > 0 else 0.0
    return utilization

def calculate_answer_completeness(response: str, query: str) -> float:
    """
    Calculate how complete the answer is relative to the query.
    
    Args:
        response (str): Generated response
        query (str): Original query
        
    Returns:
        float: Completeness score (0.0 to 1.0)
    """
    if not response or not query:
        return 0.0
    
    response_lower = response.lower()
    query_lower = query.lower()
    
    # Identify query type and expected completeness
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
    query_type = None
    
    for word in question_words:
        if word in query_lower:
            query_type = word
            break
    
    # Base completeness on response length and structure
    word_count = len(response.split())
    
    # Scoring based on response characteristics
    completeness = 0.0
    
    # Length-based scoring
    if word_count >= 50:
        completeness += 0.4
    elif word_count >= 20:
        completeness += 0.3
    elif word_count >= 10:
        completeness += 0.2
    
    # Structure-based scoring
    if '.' in response and len(response.split('.')) >= 2:  # Multiple sentences
        completeness += 0.2
    
    # Content-based scoring
    if any(keyword in response_lower for keyword in ['because', 'due to', 'therefore', 'as a result']):
        completeness += 0.1  # Explanatory content
    
    if any(keyword in response_lower for keyword in ['first', 'second', 'next', 'finally', 'steps']):
        completeness += 0.1  # Structured content
    
    # Source citation bonus
    if any(keyword in response_lower for keyword in ['source', 'according to', 'document']):
        completeness += 0.2
    
    return min(1.0, completeness)

def calculate_overall_system_accuracy(vector_store, recent_queries: List[Dict] = None) -> Dict[str, float]:
    """
    Calculate overall system accuracy metrics.
    
    Args:
        vector_store: The vector store containing embeddings
        recent_queries (List[Dict]): Recent query metrics for trend analysis
        
    Returns:
        Dict[str, float]: Overall accuracy metrics
    """
    if not hasattr(vector_store, 'chunks_data') or not vector_store.chunks_data:
        return {
            'system_health': 0.0,
            'data_quality': 0.0,
            'embedding_quality': 0.0,
            'overall_readiness': 0.0
        }
    
    chunks = vector_store.chunks_data
    
    # Calculate data quality metrics
    total_chunks = len(chunks)
    valid_chunks = sum(1 for chunk in chunks if chunk.get('content', '').strip())
    chunks_with_embeddings = sum(1 for chunk in chunks if 'embedding' in chunk)
    
    data_quality = valid_chunks / total_chunks if total_chunks > 0 else 0.0
    embedding_quality = chunks_with_embeddings / total_chunks if total_chunks > 0 else 0.0
    
    # Calculate system health based on data distribution
    chunk_sizes = [len(chunk.get('content', '')) for chunk in chunks]
    avg_chunk_size = np.mean(chunk_sizes) if chunk_sizes else 0
    
    # Ideal chunk size is between 200-800 characters
    size_score = 1.0
    if avg_chunk_size < 100 or avg_chunk_size > 1000:
        size_score = 0.6
    elif avg_chunk_size < 200 or avg_chunk_size > 800:
        size_score = 0.8
    
    system_health = (data_quality + embedding_quality + size_score) / 3
    
    # Calculate recent performance if available
    recent_performance = 0.8  # Default assumption
    if recent_queries:
        recent_accuracies = []
        for query_metrics in recent_queries[-10:]:  # Last 10 queries
            if 'retrieval' in query_metrics and 'response' in query_metrics:
                ret_acc = query_metrics['retrieval'].get('retrieval_accuracy', 0.0)
                resp_acc = query_metrics['response'].get('response_accuracy', 0.0)
                overall_acc = (ret_acc + resp_acc) / 2
                recent_accuracies.append(overall_acc)
        
        if recent_accuracies:
            recent_performance = np.mean(recent_accuracies)
    
    # Overall readiness score
    overall_readiness = (system_health + recent_performance) / 2
    
    return {
        'system_health': system_health,
        'data_quality': data_quality,
        'embedding_quality': embedding_quality,
        'overall_readiness': overall_readiness,
        'recent_performance': recent_performance
    }

def calculate_system_performance_metrics(vector_store) -> Dict[str, any]:
    """
    Calculate overall system performance metrics.
    
    Args:
        vector_store: The vector store containing embeddings
        
    Returns:
        Dict[str, any]: Dictionary containing system metrics
    """
    if not hasattr(vector_store, 'chunks_data') or not vector_store.chunks_data:
        return {
            'total_documents': 0,
            'total_chunks': 0,
            'avg_chunk_size': 0.0,
            'total_content_size': 0,
            'unique_sources': 0,
            'embedding_dimensions': 0
        }
    
    chunks = vector_store.chunks_data
    
    # Calculate metrics
    total_chunks = len(chunks)
    chunk_sizes = [len(chunk.get('content', '')) for chunk in chunks]
    total_content_size = sum(chunk_sizes)
    unique_sources = len(set(chunk.get('source', 'unknown') for chunk in chunks))
    
    # Get embedding dimensions
    embedding_dims = 0
    if chunks and 'embedding' in chunks[0]:
        embedding_dims = len(chunks[0]['embedding'])
    
    # Count unique documents
    unique_docs = len(set(chunk.get('doc_id', 'unknown') for chunk in chunks))
    
    metrics = {
        'total_documents': unique_docs,
        'total_chunks': total_chunks,
        'avg_chunk_size': np.mean(chunk_sizes) if chunk_sizes else 0.0,
        'total_content_size': total_content_size,
        'unique_sources': unique_sources,
        'embedding_dimensions': embedding_dims
    }
    
    return metrics

def format_metrics_for_display(metrics: Dict[str, any]) -> str:
    """
    Format metrics dictionary for display in Streamlit.
    
    Args:
        metrics (Dict[str, any]): Metrics dictionary
        
    Returns:
        str: Formatted metrics string
    """
    formatted_lines = []
    
    for key, value in metrics.items():
        # Format key (convert snake_case to Title Case)
        formatted_key = key.replace('_', ' ').title()
        
        # Format value based on type
        if isinstance(value, float):
            if 0 < value < 1:
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = f"{value:.2f}"
        elif isinstance(value, bool):
            formatted_value = "✅ Yes" if value else "❌ No"
        else:
            formatted_value = str(value)
        
        formatted_lines.append(f"**{formatted_key}:** {formatted_value}")
    
    return "\n".join(formatted_lines)

def safe_filename(name: str) -> str:
    # Remove/replace unsafe characters for filenames
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)

def get_summary_cache_path(doc_id: str, group_title: str, group_type: str = 'chapter') -> str:
    safe_title = safe_filename(group_title)
    return os.path.join('summaries', f'{doc_id}_{group_type}_{safe_title}.txt')

def save_summary_cache(doc_id: str, group_title: str, summary: str, group_type: str = 'chapter'):
    try:
        os.makedirs('summaries', exist_ok=True)
        path = get_summary_cache_path(doc_id, group_title, group_type)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(summary)
    except Exception as e:
        print(f'Error saving summary cache: {e}')

def load_summary_cache(doc_id: str, group_title: str, group_type: str = 'chapter') -> str:
    path = get_summary_cache_path(doc_id, group_title, group_type)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f'Error loading summary cache: {e}')
    return None

# --- Answer Caching for Procedural/Complex Q&A ---
def get_answer_cache_path(doc_id: str, query: str, section_key: str = None) -> str:
    import hashlib
    safe_query = hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]
    safe_section = section_key.replace(' ', '_') if section_key else 'nosection'
    return os.path.join('summaries', f'{doc_id}_answer_{safe_section}_{safe_query}.txt')

def save_answer_cache(doc_id: str, query: str, answer: str, section_key: str = None):
    try:
        os.makedirs('summaries', exist_ok=True)
        path = get_answer_cache_path(doc_id, query, section_key)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(answer)
    except Exception as e:
        print(f'Error saving answer cache: {e}')

def load_answer_cache(doc_id: str, query: str, section_key: str = None) -> str:
    path = get_answer_cache_path(doc_id, query, section_key)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f'Error loading answer cache: {e}')
    return None

def keyword_search(query: str, chunks: list, k: int = 10) -> list:
    """
    Perform BM25 keyword search over chunk contents.
    Returns the top-k most relevant chunks based on keyword overlap.
    Uses rank_bm25 for efficient scoring.
    Args:
        query (str): The user query string.
        chunks (list): List of chunk dicts with 'content'.
        k (int): Number of top results to return.
    Returns:
        list: Top-k most relevant chunk dicts (BM25 score > 0).
    """
    if not chunks or not query:
        return []
    # Tokenize chunk contents
    def tokenize(text):
        import re
        return [w.lower() for w in re.findall(r"\w+", text)]
    corpus = [tokenize(chunk.get('content', '')) for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_indices if scores[i] > 0]

def hybrid_search(query: str, vector_store, chunks: list, embedding_model, k_semantic: int = 10, k_keyword: int = 10, candidate_pool_size: int = 50) -> list:
    """
    Hybrid Retriever: Combines semantic (vector) and keyword (BM25) search for robust candidate retrieval.
    1. Runs semantic search (vector_store.search) for top-k_semantic results.
    2. Runs BM25 keyword search for top-k_keyword results.
    3. Merges and deduplicates candidates (by chunk_id or content hash).
    4. Returns up to candidate_pool_size unique candidates for reranking.
    Args:
        query (str): The user query string.
        vector_store: InMemoryVectorStore instance for semantic search.
        chunks (list): List of all chunk dicts.
        embedding_model: Embedding model for semantic search.
        k_semantic (int): Number of semantic results.
        k_keyword (int): Number of keyword results.
        candidate_pool_size (int): Max number of candidates to return.
    Returns:
        list: Candidate chunk dicts for reranking.
    """
    # Semantic search
    try:
        query_embedding = embedding_model.encode([query])[0]
        semantic_results = vector_store.search(query_embedding, k=k_semantic)
    except Exception:
        semantic_results = []
    # Keyword search
    keyword_results = keyword_search(query, chunks, k=k_keyword)
    # Combine and deduplicate (by chunk_id if available, else by content hash)
    seen = set()
    combined = []
    for chunk in semantic_results + keyword_results:
        chunk_id = chunk.get('chunk_id') or hash(chunk.get('content', ''))
        if chunk_id not in seen:
            combined.append(chunk)
            seen.add(chunk_id)
    # Limit to candidate_pool_size
    return combined[:candidate_pool_size]

def rewrite_query(user_query: str) -> str:
    """
    Use Gemini 1.5 Flash to rewrite or expand the user query for more effective document retrieval.
    The prompt instructs Gemini to clarify intent, add relevant keywords, and expand acronyms if obvious.
    Returns only the rewritten query as a string.
    """
    if not user_query or not isinstance(user_query, str):
        return user_query
    prompt = (
        "Rewrite or expand the following user query to be more effective for a document retrieval system. "
        "Focus on clarifying intent, adding relevant keywords, and expanding acronyms if obvious. "
        "Return only the rewritten query.\n\nUser Query: " + user_query
    )
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    response = model.generate_content(prompt)
    if response and response.text:
        return response.text.strip()
    return user_query

def compress_context(context_text: str, query: str, token_threshold: int = 50000) -> str:
    """
    Use Gemini 1.5 Flash to condense the context, extracting only information most relevant to the query.
    Preserves key facts, steps, tabular data, and page citations.
    """
    if not context_text or not query:
        return context_text
    prompt = (
        "Condense the following 'Document Context' to extract only the information most relevant to the 'User Query'. "
        "Preserve key facts, steps, and tabular data. Maintain original page citations. "
        "This compressed context will be used for final answer generation.\n\n"
        f"User Query: {query}\n\nDocument Context:\n{context_text}"
    )
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    response = model.generate_content(prompt)
    if response and response.text:
        return response.text.strip()
    return context_text

def play_audio(file_path: str):
    """
    Plays an audio file.

    Args:
        file_path (str): The path to the audio file.
    """
    if not AUDIO_PLAYBACK_AVAILABLE:
        logging.warning("Audio playback is not available. Install with: pip install playsound")
        return
    
    try:
        playsound(file_path)
    except Exception as e:
        logging.error(f"Error playing audio file {file_path}: {e}")