#!/usr/bin/env python3
"""
Index Documentation from ALL Systems
Scans documentation and config files from: Local machine, Proxmox, and Unraid
Feeds everything into ChromaDB for LLM learning

Configuration via environment variables:
  CHROMADB_HOST - ChromaDB server IP (default: <ADD-IP-ADDRESS>)
  CHROMADB_PORT - ChromaDB port (default: 8000)
  RAG_COLLECTION - Collection name (default: knowledge_base)
  RAG_EMBEDDING_MODEL - Embedding model (default: all-MiniLM-L6-v2)

Supported file types: .md, .yml, .yaml, .json, .sh, .conf, .cfg, .ini, .toml
"""

import chromadb
from chromadb.utils import embedding_functions
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import re
import json
import hashlib

# =============================================================================
# CONFIGURATION - All configurable via environment variables
# =============================================================================

def load_config():
    """Load configuration from environment variables"""
    return {
        "chromadb_host": os.getenv("CHROMADB_HOST", "<ADD-IP-ADDRESS>"),
        "chromadb_port": int(os.getenv("CHROMADB_PORT", "8000")),
        "collection_name": os.getenv("RAG_COLLECTION", "knowledge_base"),
        "embedding_model": os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "chunk_size": int(os.getenv("RAG_CHUNK_SIZE", "2000")),
        "chunk_overlap": int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
    }

CONFIG = load_config()

# File extensions to index
INDEXABLE_EXTENSIONS = {
    # Documentation
    ".md": "markdown",
    ".txt": "text",
    ".rst": "restructuredtext",

    # Configuration files
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".ini": "ini",
    ".conf": "config",
    ".cfg": "config",

    # Scripts (for learning commands/automation)
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ps1": "powershell",

    # Docker/container
    "Dockerfile": "dockerfile",
    "docker-compose.yml": "docker-compose",
    "docker-compose.yaml": "docker-compose",
}

# Systems to scan
SYSTEMS = {
    "local": {
        "enabled": True,
        "paths": [
            "~/",
            "~/Documents/",
            "~/notes/",
            "~/projects/",
        ],
        "exclude": [
            "node_modules",
            ".git",
            "venv",
            "__pycache__",
            ".cache",
            ".npm",
            "dist",
            "build",
        ]
    },
    "proxmox": {
        "enabled": True,
        "host": os.getenv("PROXMOX_HOST", "proxmox"),
        "paths": [
            "/root/",
            "/etc/pve/",
            "/usr/share/doc/",
        ],
        "exclude": [
            "*.log",
            "*.tmp",
            "*.bak",
        ]
    },
    "unraid": {
        "enabled": True,
        "host": os.getenv("UNRAID_HOST", "unraid"),
        "paths": [
            "/boot/config/",
            "/mnt/user/appdata/",
            "/mnt/user/data/docker/",
        ],
        "exclude": [
            "*.log",
            "*.tmp",
            "*.bak",
            "cache",
        ]
    }
}


def get_embedding_function():
    """Get the embedding function for ChromaDB"""
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=CONFIG["embedding_model"]
        )
    except Exception:
        return embedding_functions.DefaultEmbeddingFunction()


def get_chroma_client():
    """Connect to ChromaDB"""
    return chromadb.HttpClient(
        host=CONFIG["chromadb_host"],
        port=CONFIG["chromadb_port"]
    )


def get_file_hash(filepath):
    """Generate a consistent hash for a filepath"""
    return hashlib.md5(filepath.encode()).hexdigest()[:12]


def is_indexable_file(filepath):
    """Check if file should be indexed based on extension"""
    path = Path(filepath)

    # Check exact filename matches (Dockerfile, docker-compose.yml)
    if path.name in INDEXABLE_EXTENSIONS:
        return True

    # Check extension
    suffix = path.suffix.lower()
    return suffix in INDEXABLE_EXTENSIONS


def get_file_type(filepath):
    """Get the file type for metadata"""
    path = Path(filepath)

    if path.name in INDEXABLE_EXTENSIONS:
        return INDEXABLE_EXTENSIONS[path.name]

    suffix = path.suffix.lower()
    return INDEXABLE_EXTENSIONS.get(suffix, "unknown")


def find_files_local(paths, exclude_patterns):
    """Find all indexable files on local system"""
    found_files = []

    for path_pattern in paths:
        expanded_path = os.path.expanduser(os.path.expandvars(path_pattern))

        if not os.path.exists(expanded_path):
            print(f"   Path not found: {expanded_path}")
            continue

        for root, dirs, files in os.walk(expanded_path):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_patterns]

            for file in files:
                filepath = os.path.join(root, file)
                if is_indexable_file(filepath):
                    found_files.append(filepath)

    return found_files


def find_files_remote(host, paths, exclude_patterns):
    """Find all indexable files on remote system via SSH"""
    print(f"   Scanning {host}...")

    # Build file pattern for find command
    extensions = list(set(ext.lstrip('.') for ext in INDEXABLE_EXTENSIONS.keys()
                         if ext.startswith('.')))
    name_patterns = " -o ".join([f'-name "*.{ext}"' for ext in extensions])
    # Add exact filename matches
    exact_names = [name for name in INDEXABLE_EXTENSIONS.keys() if not name.startswith('.')]
    if exact_names:
        name_patterns += " -o " + " -o ".join([f'-name "{name}"' for name in exact_names])

    exclude_args = " ".join([f'-not -path "*/{pattern}/*"' for pattern in exclude_patterns
                             if not pattern.startswith('*')])

    found_files = []

    for path in paths:
        find_cmd = f'find {path} -type f \\( {name_patterns} \\) {exclude_args} 2>/dev/null'
        ssh_cmd = ["ssh", host, find_cmd]

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
                found_files.extend(files)
        except subprocess.TimeoutExpired:
            print(f"   Timeout scanning {path} on {host}")
        except Exception as e:
            print(f"   Error scanning {path} on {host}: {e}")

    return found_files


def copy_remote_file(host, remote_path, local_temp_dir):
    """Copy remote file to local temp directory via SSH"""
    safe_filename = remote_path.replace('/', '_')
    local_path = os.path.join(local_temp_dir, safe_filename)

    try:
        scp_cmd = ["scp", "-q", f"{host}:{remote_path}", local_path]
        result = subprocess.run(
            scp_cmd,
            capture_output=True,
            timeout=30
        )

        if result.returncode == 0:
            return local_path
        return None
    except Exception as e:
        print(f"   Failed to copy {remote_path}: {e}")
        return None


def parse_config_file(content, file_type):
    """Parse config files to extract meaningful content"""
    if file_type == "json":
        try:
            data = json.loads(content)
            # Pretty print for better chunking
            return json.dumps(data, indent=2)
        except:
            return content

    elif file_type in ("yaml", "docker-compose"):
        # YAML is already readable, just clean up
        return content

    elif file_type == "shell":
        # Extract comments and important lines
        lines = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#') or line.startswith('export ') or \
               '=' in line or line.startswith('function ') or \
               any(cmd in line for cmd in ['docker', 'ssh', 'curl', 'wget', 'apt', 'yum']):
                lines.append(line)
        return '\n'.join(lines) if lines else content

    return content


def extract_metadata_from_file(content, filepath, system="local"):
    """Extract metadata from file content"""
    path = Path(filepath)
    file_type = get_file_type(filepath)

    metadata = {
        "source": path.stem,
        "filename": path.name,
        "filepath": filepath,
        "system": system,
        "file_type": file_type,
        "indexed_at": datetime.now().isoformat(),
    }

    # Extract title from markdown
    if file_type == "markdown":
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()

    # Infer category
    filename_lower = path.stem.lower()
    filepath_lower = filepath.lower()

    if any(word in filename_lower for word in ['setup', 'install', 'guide', 'howto']):
        metadata["category"] = "guide"
    elif any(word in filename_lower for word in ['troubleshoot', 'debug', 'fix', 'error']):
        metadata["category"] = "troubleshooting"
    elif file_type in ('config', 'yaml', 'json', 'ini', 'toml'):
        metadata["category"] = "configuration"
    elif file_type in ('shell', 'powershell'):
        metadata["category"] = "script"
    elif file_type == 'docker-compose':
        metadata["category"] = "docker"
    elif any(word in filename_lower for word in ['readme', 'overview', 'intro']):
        metadata["category"] = "overview"
    else:
        metadata["category"] = "documentation"

    # Extract topics
    topics = []
    content_lower = content.lower()

    topic_keywords = {
        'docker': ['docker', 'container', 'compose'],
        'proxmox': ['proxmox', 'pve', 'lxc', 'kvm', 'qemu'],
        'unraid': ['unraid', 'array', 'parity'],
        'ollama': ['ollama', 'llm', 'mistral', 'llama'],
        'chromadb': ['chromadb', 'chroma', 'vector'],
        'gpu': ['gpu', 'passthrough', 'rocm', 'cuda', 'nvidia'],
        'networking': ['network', 'vlan', 'firewall', 'vpn', 'dns', 'nginx'],
        'storage': ['storage', 'disk', 'raid', 'zfs', 'nfs', 'smb'],
        'backup': ['backup', 'restore', 'snapshot', 'rsync'],
        'monitoring': ['monitoring', 'grafana', 'prometheus', 'telegraf'],
        'automation': ['automation', 'script', 'ansible', 'cron'],
        'security': ['security', 'ssl', 'certificate', 'auth', 'tls'],
        'plex': ['plex', 'media', 'transcode'],
    }

    for topic, keywords in topic_keywords.items():
        if any(kw in content_lower or kw in filepath_lower for kw in keywords):
            topics.append(topic)

    if topics:
        metadata["topics"] = ",".join(topics)

    return metadata


def chunk_document(content, chunk_size=None, overlap=None):
    """Split document into overlapping chunks"""
    if chunk_size is None:
        chunk_size = CONFIG["chunk_size"]
    if overlap is None:
        overlap = CONFIG["chunk_overlap"]

    # Try to split on section boundaries for markdown
    section_pattern = r'(^##?\s+.+$)'
    sections = re.split(section_pattern, content, flags=re.MULTILINE)

    chunks = []
    current_chunk = ""

    for section in sections:
        if len(current_chunk) + len(section) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:] + section if overlap > 0 else section
        else:
            current_chunk += section

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Fallback to simple chunking if needed
    if not chunks or max(len(c) for c in chunks) > chunk_size * 1.5:
        chunks = []
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())

    return chunks if chunks else [content]


def delete_existing_documents(collection, filepath):
    """Delete existing documents for a filepath before re-indexing"""
    try:
        # Get file hash for ID matching
        file_hash = get_file_hash(filepath)

        # Try to find and delete existing documents
        results = collection.get(
            where={"filepath": filepath},
            include=[]
        )

        if results and results['ids']:
            collection.delete(ids=results['ids'])
            return len(results['ids'])
    except Exception:
        pass

    return 0


def index_file(collection, filepath, system="local", verbose=True):
    """Index a single file with deduplication"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if not content.strip() or len(content) < 50:
            return 0

        file_type = get_file_type(filepath)

        # Parse config files for better content
        if file_type in ('json', 'yaml', 'shell', 'docker-compose'):
            content = parse_config_file(content, file_type)

        # Extract metadata
        metadata = extract_metadata_from_file(content, filepath, system)

        # Delete existing documents for this file (deduplication)
        deleted = delete_existing_documents(collection, filepath)
        if deleted > 0 and verbose:
            print(f"   (updated - removed {deleted} old chunks)")

        # Chunk document
        chunks = chunk_document(content)

        if verbose:
            filename = Path(filepath).name
            file_type_str = metadata.get('file_type', '?')
            print(f"   {filename[:45]:<45} {len(chunks):>2} chunks [{system}/{file_type_str}]")

        # Prepare documents for batch insert
        documents = []
        metadatas = []
        ids = []

        file_hash = get_file_hash(filepath)

        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)

            # Create unique, reproducible ID
            doc_id = f"{system}-{file_hash}-{i}"

            documents.append(chunk)
            metadatas.append(chunk_metadata)
            ids.append(doc_id)

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        return len(chunks)

    except Exception as e:
        if verbose:
            print(f"   Error indexing {filepath}: {e}")
        return 0


def scan_and_index_system(collection, system_name, config, temp_dir=None, verbose=True):
    """Scan and index all files from a system"""
    if not config.get("enabled", False):
        return {"files": 0, "chunks": 0, "skipped": 0}

    print(f"\n{'='*60}")
    print(f"SCANNING: {system_name.upper()}")
    print(f"{'='*60}")

    stats = {"files": 0, "chunks": 0, "skipped": 0}

    if system_name == "local":
        print("Searching for indexable files...")
        files = find_files_local(
            config["paths"],
            config.get("exclude", [])
        )

        print(f"Found {len(files)} indexable files\n")

        for filepath in files:
            chunks = index_file(collection, filepath, system_name, verbose)
            if chunks > 0:
                stats["files"] += 1
                stats["chunks"] += chunks
            else:
                stats["skipped"] += 1

    else:
        host = config["host"]
        print(f"Connecting to {host}...")

        remote_files = find_files_remote(
            host,
            config["paths"],
            config.get("exclude", [])
        )

        print(f"Found {len(remote_files)} indexable files on {host}\n")

        if not remote_files:
            return stats

        for remote_path in remote_files:
            local_path = copy_remote_file(host, remote_path, temp_dir)

            if local_path:
                # Use remote path for metadata, local path for reading
                chunks = index_file(collection, local_path, system_name, verbose)

                # Update the metadata filepath to remote path
                if chunks > 0:
                    stats["files"] += 1
                    stats["chunks"] += chunks
                else:
                    stats["skipped"] += 1

                try:
                    os.remove(local_path)
                except:
                    pass
            else:
                stats["skipped"] += 1

    return stats


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Index documentation and config files from all systems into ChromaDB"
    )
    parser.add_argument("--local-only", action="store_true", help="Only scan local files")
    parser.add_argument("--proxmox-only", action="store_true", help="Only scan Proxmox")
    parser.add_argument("--unraid-only", action="store_true", help="Only scan Unraid")
    parser.add_argument("--clear", action="store_true", help="Clear collection before indexing")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be indexed")
    parser.add_argument("--config", action="store_true", help="Show configuration")

    args = parser.parse_args()

    print("="*60)
    print("MULTI-SYSTEM DOCUMENTATION INDEXER")
    print("Index docs & configs from Local, Proxmox, and Unraid")
    print("="*60)

    if args.config:
        print("\nConfiguration:")
        print(f"  ChromaDB:   {CONFIG['chromadb_host']}:{CONFIG['chromadb_port']}")
        print(f"  Collection: {CONFIG['collection_name']}")
        print(f"  Embedding:  {CONFIG['embedding_model']}")
        print(f"  Chunk size: {CONFIG['chunk_size']} (overlap: {CONFIG['chunk_overlap']})")
        print(f"\nIndexable extensions:")
        for ext, ftype in sorted(INDEXABLE_EXTENSIONS.items()):
            print(f"  {ext:<20} -> {ftype}")
        print(f"\nSystems:")
        for name, cfg in SYSTEMS.items():
            status = "enabled" if cfg.get("enabled") else "disabled"
            host = cfg.get("host", "localhost")
            print(f"  {name:<10} [{status}] {host}")
        return 0

    # Connect to ChromaDB
    print(f"\nConnecting to ChromaDB at {CONFIG['chromadb_host']}:{CONFIG['chromadb_port']}...")
    client = get_chroma_client()

    # Get or create collection
    try:
        if args.clear:
            try:
                client.delete_collection(name=CONFIG["collection_name"])
                print(f"Cleared existing collection '{CONFIG['collection_name']}'")
            except:
                pass

        collection = client.get_or_create_collection(
            name=CONFIG["collection_name"],
            embedding_function=get_embedding_function()
        )
        current_count = collection.count()
        print(f"Connected to collection '{CONFIG['collection_name']}'")
        print(f"  Current documents: {current_count}")
    except Exception as e:
        print(f"Failed to connect to ChromaDB: {e}")
        return 1

    if args.dry_run:
        print("\nDRY RUN MODE - Nothing will be indexed")

    # Create temp directory for remote files
    temp_dir = tempfile.mkdtemp(prefix="rag-index-")

    try:
        scan_all = not (args.local_only or args.proxmox_only or args.unraid_only)

        total_stats = {"files": 0, "chunks": 0, "skipped": 0}

        for system_name, config in SYSTEMS.items():
            should_scan = (
                scan_all or
                (args.local_only and system_name == "local") or
                (args.proxmox_only and system_name == "proxmox") or
                (args.unraid_only and system_name == "unraid")
            )

            if should_scan and not args.dry_run:
                stats = scan_and_index_system(
                    collection,
                    system_name,
                    config,
                    temp_dir,
                    verbose=not args.quiet
                )

                total_stats["files"] += stats["files"]
                total_stats["chunks"] += stats["chunks"]
                total_stats["skipped"] += stats["skipped"]

                print(f"\n{system_name.upper()} Summary:")
                print(f"  Files indexed: {stats['files']}")
                print(f"  Chunks created: {stats['chunks']}")
                print(f"  Files skipped: {stats['skipped']}")

        # Final summary
        new_count = collection.count()

        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total files indexed:     {total_stats['files']}")
        print(f"Total chunks created:    {total_stats['chunks']}")
        print(f"Files skipped:           {total_stats['skipped']}")
        print(f"Documents before:        {current_count}")
        print(f"Documents after:         {new_count}")
        print(f"Net change:              {new_count - current_count:+d}")
        print(f"\nYour RAG system now has documentation from all systems!")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    exit(main())
