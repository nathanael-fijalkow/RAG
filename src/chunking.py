from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import re
from pathlib import Path
from rich import print

from .llm import LLM

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, str]

    def pretty_print(self):
        excerpt = (self.text[:200] + "...") if len(self.text) > 200 else self.text
        return f"{self.metadata.get('source')}#page={self.metadata.get('page')}\n> {excerpt}"

def chunk_fixed(text: str, metadata: Dict[str, str], size: int = 600, overlap: int = 100) -> List[Chunk]:
    # print(f"[chunk_fixed] Chunking text of length {len(text)} with size={size}, overlap={overlap}")
    chunks: List[Chunk] = []
    for start in range(0, len(text), size - overlap):
        end = min(len(text), start + size)
        print(f"[chunk_fixed] Creating chunk from {start} to {end}")
        chunks.append(Chunk(text=text[start:end], metadata=dict(metadata, chunk_strategy="fixed")))
    return chunks

def chunk_recursive(text: str, metadata: Dict[str, str], target: int = 700) -> List[Chunk]:
    # simple heuristic: split by paragraphs -> sentences
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: List[Chunk] = []
    buf = ""
    for para in paras:
        if len(buf) + len(para) + 2 <= target:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                out.append(Chunk(text=buf, metadata=dict(metadata, chunk_strategy="recursive")))
            if len(para) <= target:
                out.append(Chunk(text=para, metadata=dict(metadata, chunk_strategy="recursive")))
            else:
                # sentence split
                sentences = [s.strip() for s in para.replace("?", ".").replace("!", ".").split(".") if s.strip()]
                buf2 = ""
                for s in sentences:
                    if len(buf2) + len(s) + 2 <= target:
                        buf2 = f"{buf2}. {s}" if buf2 else s
                    else:
                        if buf2:
                            out.append(Chunk(text=buf2, metadata=dict(metadata, chunk_strategy="recursive")))
                        buf2 = s
                if buf2:
                    out.append(Chunk(text=buf2, metadata=dict(metadata, chunk_strategy="recursive")))
            buf = ""
    if buf:
        out.append(Chunk(text=buf, metadata=dict(metadata, chunk_strategy="recursive")))
    return out


def chunk_semantic(text: str, metadata: Dict[str, str], use_llm_summary: bool = True, max_chunk_size: int = 2000) -> List[Chunk]:
    """
    Split text based on semantic sections marked by markdown headers:
    - # H1, ## H2, or == title == (major sections)
    - ### H3, #### H4, or === subtitle === (minor sections)
    
    Handles both Wikipedia format and FineWiki markdown format.
    Removes HTML/Markdown comments before chunking.
    Each chunk is prefixed with filename, title hierarchy, and optionally an LLM-generated summary.
    """
    
    # Extract filename from source metadata
    source = metadata.get('source', 'unknown')
    filename = Path(source).stem if source else 'unknown'
    
    # Remove HTML and Markdown comments
    # Remove HTML comments: <!-- ... -->
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove Markdown reference-style links definitions: [id]: url "title"
    text = re.sub(r'^\[.+?\]:\s+.+$', '', text, flags=re.MULTILINE)
    
    # Parse the document into sections
    sections = []
    current_title = None
    current_subtitle = None
    current_content = []
    
    lines = text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()
        
        # Check for major section headers
        # Markdown H1: # Title
        h1_match = re.match(r'^#\s+(.+)$', line_stripped)
        # Markdown H2: ## Title
        h2_match = re.match(r'^##\s+(.+)$', line_stripped)
        # Wikipedia/custom format: == Title ==
        eq2_match = re.match(r'^==\s+(.+?)\s+==$', line_stripped)
        
        if h1_match or h2_match or eq2_match:
            # Save previous section if exists
            if current_content:
                sections.append({
                    'title': current_title,
                    'subtitle': current_subtitle,
                    'content': '\n'.join(current_content).strip()
                })
                current_content = []
            
            # Extract title text
            if h1_match:
                current_title = h1_match.group(1).strip()
            elif h2_match:
                current_title = h2_match.group(1).strip()
            else:  # eq2_match
                current_title = eq2_match.group(1).strip()
            
            current_subtitle = None
            i += 1
            continue
        
        # Check for minor section headers (subsections)
        # Markdown H3: ### Subtitle
        h3_match = re.match(r'^###\s+(.+)$', line_stripped)
        # Markdown H4: #### Subtitle
        h4_match = re.match(r'^####\s+(.+)$', line_stripped)
        # Wikipedia/custom format: === Subtitle === or ==== Subtitle ====
        eq3_match = re.match(r'^={3,4}\s+(.+?)\s+={3,4}$', line_stripped)
        
        if h3_match or h4_match or eq3_match:
            # Save previous subsection if exists
            if current_content:
                sections.append({
                    'title': current_title,
                    'subtitle': current_subtitle,
                    'content': '\n'.join(current_content).strip()
                })
                current_content = []
            
            # Extract subtitle text
            if h3_match:
                current_subtitle = h3_match.group(1).strip()
            elif h4_match:
                current_subtitle = h4_match.group(1).strip()
            else:  # eq3_match
                current_subtitle = eq3_match.group(1).strip()
            
            i += 1
            continue
        
        # Regular content line
        current_content.append(line)
        i += 1
    
    # Save last section
    if current_content:
        sections.append({
            'title': current_title,
            'subtitle': current_subtitle,
            'content': '\n'.join(current_content).strip()
        })
    
    # Create chunks with context headers
    chunks: List[Chunk] = []
    
    for idx, section in enumerate(sections):
        if not section['content']:
            continue
        
        # Build context header
        header_parts = [f"Document: {filename}"]
        
        if section['title']:
            header_parts.append(f"Section: {section['title']}")
        
        if section['subtitle']:
            header_parts.append(f"Subsection: {section['subtitle']}")
        
        header = ' | '.join(header_parts)
        
        # Optionally add LLM summary
        summary = ""
        if use_llm_summary and len(section['content']) > 200:
            try:
                llm = LLM()
                hierarchy = f"{section['title'] or 'Document'}"
                if section['subtitle']:
                    hierarchy += f" > {section['subtitle']}"
                
                summary_prompt = f"Provide a one-sentence summary (max 50 words) of this section:\n\n{section['content'][:500]}"
                summary_text = llm.generate(
                    summary_prompt,
                    system="You are a helpful assistant that creates concise summaries. Output only the summary, no preamble."
                )
                summary = f"\nSummary: {summary_text}\n"
                print(f"[chunk_semantic] Generated summary for {hierarchy}")
            except Exception as e:
                print(f"[chunk_semantic] Warning: Could not generate summary: {e}")
                summary = ""
        
        # Split large sections if needed
        content = section['content']
        if len(content) <= max_chunk_size:
            # Single chunk
            chunk_text = f"{header}{summary}\n\n{content}"
            chunk_metadata = dict(
                metadata,
                chunk_strategy="semantic",
                section_title=section['title'] or "",
                section_subtitle=section['subtitle'] or "",
                chunk_index=str(idx)
            )
            chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))
        else:
            # Split into smaller chunks while preserving header
            num_parts = (len(content) + max_chunk_size - 1) // max_chunk_size
            for part_idx in range(num_parts):
                start = part_idx * max_chunk_size
                end = min((part_idx + 1) * max_chunk_size, len(content))
                part_content = content[start:end]
                
                chunk_text = f"{header} (part {part_idx + 1}/{num_parts}){summary}\n\n{part_content}"
                chunk_metadata = dict(
                    metadata,
                    chunk_strategy="semantic",
                    section_title=section['title'] or "",
                    section_subtitle=section['subtitle'] or "",
                    chunk_index=f"{idx}-{part_idx}"
                )
                chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))
    
    print(f"[chunk_semantic] Created {len(chunks)} chunks from {len(sections)} sections")
    return chunks
