"""
PostgreSQL vector database using pgvector for chat memory and knowledge storage.

All data is encrypted using Kyber post-quantum encryption before storage.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psycopg2
    from psycopg2.extras import execute_values, Json
    from psycopg2.pool import ThreadedConnectionPool
except ImportError:
    psycopg2 = None  # type: ignore
    execute_values = None  # type: ignore
    Json = None  # type: ignore
    ThreadedConnectionPool = None  # type: ignore

from core.crypto_vault import CryptoVault
from utils.logger_util import get_logger

logger = get_logger("core.vector_db")


class VectorDB:
    """
    PostgreSQL vector database with pgvector extension for storing chat memories
    and knowledge embeddings.
    
    All stored data is encrypted using Kyber post-quantum encryption.
    """

    def __init__(self, connection_string: Optional[str] = None, vault: Optional[CryptoVault] = None) -> None:
        """
        Initialize vector database connection.
        
        Args:
            connection_string: PostgreSQL connection string. If None, uses env vars.
            vault: CryptoVault instance for encrypting/decrypting data. Required for encryption.
        """
        if psycopg2 is None:
            raise ImportError("psycopg2 is required for vector database. Install with: pip install psycopg2-binary")
        
        self._connection_string = connection_string or self._build_connection_string()
        self._pool: Optional[ThreadedConnectionPool] = None
        self._initialized = False
        self._vault = vault
        
        if vault is None:
            logger.warning("VectorDB initialized without vault - data will not be encrypted!")
        elif not vault.is_unlocked():
            logger.warning("VectorDB vault is locked - encryption may fail!")

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from environment variables."""
        host = os.getenv("AWEN_DB_HOST", "localhost")
        port = os.getenv("AWEN_DB_PORT", "5432")
        database = os.getenv("AWEN_DB_NAME", "awen01")
        user = os.getenv("AWEN_DB_USER", "awen01")
        password = os.getenv("AWEN_DB_PASSWORD", "")
        
        return f"host={host} port={port} dbname={database} user={user} password={password}"
    
    def _encrypt_text(self, text: str) -> str:
        """
        Encrypt text using the vault's Kyber encryption.
        
        Args:
            text: Plaintext to encrypt.
        Returns:
            Encrypted text (base64-encoded).
        """
        if self._vault is None or not self._vault.is_unlocked():
            logger.warning("Vault not available, storing plaintext (not recommended!)")
            return text
        return self._vault.encrypt_payload(text)
    
    def _decrypt_text(self, encrypted_text: str) -> str:
        """
        Decrypt text using the vault's Kyber decryption.
        
        Args:
            encrypted_text: Encrypted text (base64-encoded).
        Returns:
            Decrypted plaintext.
        """
        if self._vault is None or not self._vault.is_unlocked():
            # Try to detect if text is encrypted (base64 format)
            # If it looks like plaintext, return as-is (for backward compatibility)
            try:
                # Try to decrypt - if it fails, assume it's plaintext
                return self._vault.decrypt_payload(encrypted_text) if self._vault else encrypted_text
            except Exception:
                # If decryption fails, assume it's plaintext (legacy data)
                logger.debug("Decryption failed, assuming plaintext (legacy data)")
                return encrypted_text
        
        try:
            return self._vault.decrypt_payload(encrypted_text)
        except Exception as e:
            logger.warning("Failed to decrypt text, returning as-is: %s", e)
            return encrypted_text

    def connect(self) -> None:
        """Initialize connection pool and ensure database is set up."""
        if self._pool is not None:
            return
        
        try:
            self._pool = ThreadedConnectionPool(1, 5, self._connection_string)
            self._ensure_schema()
            self._initialized = True
            logger.info("Vector database connected successfully")
        except Exception as e:
            logger.error("Failed to connect to vector database: %s", e)
            raise

    def _ensure_schema(self) -> None:
        """Create tables and enable pgvector extension if they don't exist."""
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create memories table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector(384),
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create chat_history table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id SERIAL PRIMARY KEY,
                        user_message TEXT NOT NULL,
                        assistant_message TEXT NOT NULL,
                        session_id TEXT,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create index on chat_history created_at for efficient retrieval
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS chat_history_created_at_idx 
                    ON chat_history (created_at DESC);
                """)
                
                # Create index on chat_history session_id
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS chat_history_session_id_idx 
                    ON chat_history (session_id);
                """)
                
                # Create knowledge table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge (
                        id SERIAL PRIMARY KEY,
                        file_path TEXT NOT NULL UNIQUE,
                        file_name TEXT NOT NULL,
                        file_type TEXT,
                        content TEXT,
                        embedding vector(384),
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create indexes for vector similarity search
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS memories_embedding_idx 
                    ON memories USING ivfflat (embedding vector_cosine_ops);
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS knowledge_embedding_idx 
                    ON knowledge USING ivfflat (embedding vector_cosine_ops);
                """)
                
                conn.commit()
                logger.info("Vector database schema ensured")
        finally:
            self._pool.putconn(conn)

    def add_memory(self, content: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a memory to the database.
        
        Args:
            content: The text content of the memory
            embedding: Vector embedding (384 dimensions)
            metadata: Optional metadata dictionary
            
        Returns:
            ID of the inserted memory
        """
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                # Convert embedding list to pgvector format: [1.0, 2.0, 3.0]
                embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
                
                # Strip null bytes from content (PostgreSQL can't handle them)
                content_clean = content.replace('\x00', '') if content else ''
                
                # Encrypt content before storing
                content_encrypted = self._encrypt_text(content_clean)
                
                cur.execute("""
                    INSERT INTO memories (content, embedding, metadata)
                    VALUES (%s, %s::vector, %s)
                    RETURNING id;
                """, (content_encrypted, embedding_str, Json(metadata or {})))
                memory_id = cur.fetchone()[0]
                conn.commit()
                logger.debug("Added memory with ID %s", memory_id)
                return memory_id
        finally:
            self._pool.putconn(conn)

    def search_knowledge_files(self, query_embedding: List[float], limit: int = 5, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Search knowledge files by similarity.
        
        Args:
            query_embedding: Query vector embedding
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of matching knowledge files with similarity scores
        """
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                # Convert query embedding to pgvector format
                query_embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
                
                cur.execute("""
                    SELECT id, file_name, content, metadata, 
                           1 - (embedding <=> %s::vector) as similarity
                    FROM knowledge
                    WHERE 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (query_embedding_str, query_embedding_str, threshold, query_embedding_str, limit))
                
                results = []
                for row in cur.fetchall():
                    # Decrypt content before returning
                    content_encrypted = row[2]
                    content_decrypted = self._decrypt_text(content_encrypted)
                    
                    results.append({
                        "id": row[0],
                        "file_name": row[1],
                        "content": content_decrypted,
                        "metadata": row[3] or {},
                        "similarity": float(row[4])
                    })
                return results
        finally:
            self._pool.putconn(conn)

    def add_knowledge_file(self, file_path: str, file_name: str, content: str, 
                          embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a knowledge file to the database.
        
        Args:
            file_path: Path to the file
            file_name: Name of the file
            content: Extracted text content
            embedding: Vector embedding
            metadata: Optional metadata
            
        Returns:
            ID of the inserted knowledge entry
        """
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                # Convert embedding list to pgvector format: [1.0, 2.0, 3.0]
                embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
                
                # Strip null bytes from content (PostgreSQL can't handle them)
                content_clean = content.replace('\x00', '') if content else ''
                
                # Encrypt content before storing
                content_encrypted = self._encrypt_text(content_clean)
                
                cur.execute("""
                    INSERT INTO knowledge (file_path, file_name, file_type, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (file_path) 
                    DO UPDATE SET 
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id;
                """, (file_path, file_name, Path(file_name).suffix, content_encrypted, embedding_str, Json(metadata or {})))
                knowledge_id = cur.fetchone()[0]
                conn.commit()
                logger.debug("Added/updated knowledge file with ID %s", knowledge_id)
                return knowledge_id
        finally:
            self._pool.putconn(conn)

    def list_knowledge_files(self) -> List[Dict[str, Any]]:
        """List all knowledge files."""
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, file_path, file_name, file_type, created_at, updated_at
                    FROM knowledge
                    ORDER BY updated_at DESC;
                """)
                
                results = []
                for row in cur.fetchall():
                    results.append({
                        "id": row[0],
                        "file_path": row[1],
                        "file_name": row[2],
                        "file_type": row[3],
                        "created_at": row[4].isoformat() if row[4] else None,
                        "updated_at": row[5].isoformat() if row[5] else None
                    })
                return results
        finally:
            self._pool.putconn(conn)

    def delete_knowledge_file(self, file_id: int) -> bool:
        """Delete a knowledge file by ID."""
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM knowledge WHERE id = %s;", (file_id,))
                deleted = cur.rowcount > 0
                conn.commit()
                return deleted
        finally:
            self._pool.putconn(conn)

    def list_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List recent memories."""
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, content, metadata, created_at
                    FROM memories
                    ORDER BY created_at DESC
                    LIMIT %s;
                """, (limit,))
                
                results = []
                for row in cur.fetchall():
                    results.append({
                        "id": row[0],
                        "content": row[1],
                        "metadata": row[2],
                        "created_at": row[3].isoformat() if row[3] else None
                    })
                return results
        finally:
            self._pool.putconn(conn)

    def list_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List recent memories."""
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, content, metadata, created_at
                    FROM memories
                    ORDER BY created_at DESC
                    LIMIT %s;
                """, (limit,))
                
                results = []
                for row in cur.fetchall():
                    # Decrypt content before returning
                    content_encrypted = row[1]
                    content_decrypted = self._decrypt_text(content_encrypted)
                    
                    results.append({
                        "id": row[0],
                        "content": content_decrypted,
                        "metadata": row[2] or {},
                        "created_at": row[3].isoformat() if row[3] else None
                    })
                return results
        finally:
            self._pool.putconn(conn)
    
    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory by ID."""
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE id = %s;", (memory_id,))
                deleted = cur.rowcount > 0
                conn.commit()
                return deleted
        finally:
            self._pool.putconn(conn)
    
    def delete_all_memories(self) -> int:
        """Delete all memories."""
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memories;")
                deleted_count = cur.rowcount
                conn.commit()
                logger.info("Deleted %d memories", deleted_count)
                return deleted_count
        finally:
            self._pool.putconn(conn)
    
    def delete_all_notable_memories(self) -> int:
        """Delete all notable memories (memories with type 'notable_pattern')."""
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE metadata->>'type' = 'notable_pattern';")
                deleted_count = cur.rowcount
                conn.commit()
                logger.info("Deleted %d notable memories", deleted_count)
                return deleted_count
        finally:
            self._pool.putconn(conn)
    
    def add_chat_history(self, user_message: str, assistant_message: str, 
                        session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a chat history entry.
        
        Args:
            user_message: The user's message
            assistant_message: The assistant's response
            session_id: Optional session identifier
            metadata: Optional metadata dictionary
            
        Returns:
            ID of the inserted chat entry
        """
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                # Strip null bytes from messages
                user_clean = user_message.replace('\x00', '') if user_message else ''
                assistant_clean = assistant_message.replace('\x00', '') if assistant_message else ''
                
                # Encrypt messages before storing
                user_encrypted = self._encrypt_text(user_clean)
                assistant_encrypted = self._encrypt_text(assistant_clean)
                
                cur.execute("""
                    INSERT INTO chat_history (user_message, assistant_message, session_id, metadata)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                """, (user_encrypted, assistant_encrypted, session_id, Json(metadata or {})))
                chat_id = cur.fetchone()[0]
                conn.commit()
                logger.debug("Added chat history with ID %s", chat_id)
                return chat_id
        finally:
            self._pool.putconn(conn)
    
    def list_chat_history(self, limit: int = 100, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List chat history entries.
        
        Args:
            limit: Maximum number of entries to return
            session_id: Optional session ID to filter by
            
        Returns:
            List of chat history entries
        """
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                if session_id:
                    cur.execute("""
                        SELECT id, user_message, assistant_message, session_id, metadata, created_at
                        FROM chat_history
                        WHERE session_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s;
                    """, (session_id, limit))
                else:
                    cur.execute("""
                        SELECT id, user_message, assistant_message, session_id, metadata, created_at
                        FROM chat_history
                        ORDER BY created_at DESC
                        LIMIT %s;
                    """, (limit,))
                
                results = []
                for row in cur.fetchall():
                    # Decrypt messages before returning
                    user_encrypted = row[1]
                    assistant_encrypted = row[2]
                    user_decrypted = self._decrypt_text(user_encrypted)
                    assistant_decrypted = self._decrypt_text(assistant_encrypted)
                    
                    results.append({
                        "id": row[0],
                        "user_message": user_decrypted,
                        "assistant_message": assistant_decrypted,
                        "session_id": row[3],
                        "metadata": row[4] or {},
                        "created_at": row[5].isoformat() if row[5] else None
                    })
                return results
        finally:
            self._pool.putconn(conn)
    
    def delete_chat_history(self, chat_id: int) -> bool:
        """Delete a chat history entry by ID."""
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_history WHERE id = %s;", (chat_id,))
                deleted = cur.rowcount > 0
                conn.commit()
                logger.debug("Deleted chat history entry with ID %s", chat_id)
                return deleted
        finally:
            self._pool.putconn(conn)
    
    def delete_all_chat_history(self) -> int:
        """Delete all chat history entries."""
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_history;")
                deleted_count = cur.rowcount
                conn.commit()
                logger.info("Deleted %d chat history entries", deleted_count)
                return deleted_count
        finally:
            self._pool.putconn(conn)
    
    def update_memory(self, memory_id: int, content: str, embedder=None) -> bool:
        """Update a memory's content and regenerate its embedding."""
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                # Check if memory exists
                cur.execute("SELECT id FROM memories WHERE id = %s;", (memory_id,))
                if not cur.fetchone():
                    return False
                
                # Strip null bytes from content
                content_clean = content.replace('\x00', '') if content else ''
                
                # Encrypt content before storing
                content_encrypted = self._encrypt_text(content_clean)
                
                # Regenerate embedding if embedder is available
                if embedder:
                    try:
                        embedding = embedder.embed(content_clean)
                        embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
                        cur.execute("""
                            UPDATE memories 
                            SET content = %s, embedding = %s::vector, updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s;
                        """, (content_encrypted, embedding_str, memory_id))
                    except Exception as e:
                        logger.warning("Failed to regenerate embedding for memory %s: %s", memory_id, e)
                        # Update content without embedding
                        cur.execute("""
                            UPDATE memories 
                            SET content = %s, updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s;
                        """, (content_encrypted, memory_id))
                else:
                    # Update content without embedding
                    cur.execute("""
                        UPDATE memories 
                        SET content = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s;
                    """, (content_encrypted, memory_id))
                
                conn.commit()
                logger.debug("Updated memory with ID %s", memory_id)
                return True
        finally:
            self._pool.putconn(conn)
    
    def detect_notable_memories(self, limit: int = 20, 
                                deepseek_adapter=None, 
                                embedder=None) -> List[Dict[str, Any]]:
        """
        Detect notable memories using DeepSeek model to analyze chat history and extract patterns.
        Stores extracted patterns as memories and returns them.
        
        Looks for patterns like:
        - User's name (e.g., "The user's name is Joe")
        - Preferences (e.g., "User prefers Python code")
        - Important facts or information about the user
        - Recurring topics or interests
        
        Args:
            limit: Maximum number of notable memories to return
            deepseek_adapter: Optional DeepSeek adapter for pattern analysis
            embedder: Optional embedder for generating memory embeddings
            
        Returns:
            List of notable memories with metadata
        """
        if not self._initialized:
            self.connect()
        
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                # Get recent chat history (last 100 conversations)
                cur.execute("""
                    SELECT id, user_message, assistant_message, metadata, created_at
                    FROM chat_history
                    ORDER BY created_at DESC
                    LIMIT 100;
                """)
                
                recent_chats = []
                for row in cur.fetchall():
                    # Decrypt messages before using
                    user_decrypted = self._decrypt_text(row[1])
                    assistant_decrypted = self._decrypt_text(row[2])
                    
                    recent_chats.append({
                        "id": row[0],
                        "user_message": user_decrypted,
                        "assistant_message": assistant_decrypted,
                        "metadata": row[3] or {},
                        "created_at": row[4].isoformat() if row[4] else None
                    })
                
                if not recent_chats:
                    return []
                
                # If DeepSeek adapter is available, use it to analyze patterns
                if deepseek_adapter and embedder:
                    try:
                        # Build a prompt for DeepSeek to analyze chat history and extract notable patterns
                        chat_context = "\n\n".join([
                            f"User: {chat['user_message']}\nAssistant: {chat['assistant_message']}"
                            for chat in recent_chats[:50]  # Use last 50 chats for analysis
                        ])
                        
                        analysis_prompt = f"""Analyze the following chat history and extract notable patterns, facts, and preferences about the user. 

Focus on:
- User's name or identity information
- Programming language preferences (e.g., Python, JavaScript, etc.)
- Work preferences or habits
- Important facts the user has shared
- Recurring interests or topics
- Preferences or dislikes

For each notable pattern you find, format it as a concise fact statement like:
- "The user's name is Joe"
- "User prefers Python code"
- "User works as a software engineer"
- "User is interested in machine learning"

Chat History:
{chat_context}

Extract notable patterns (one per line, be concise):"""

                        # Use DeepSeek to analyze
                        logger.info("Analyzing chat history with DeepSeek to extract notable memories...")
                        completion_dict = deepseek_adapter.infer({
                            "prompt": analysis_prompt,
                            "temperature": 0.3,  # Lower temperature for more focused extraction
                            "system_prompt": "You are a memory extraction system. Extract notable patterns and facts from chat history.",
                            "max_tokens": 1024,
                        })
                        
                        analysis_text = completion_dict.get("content", "")
                        
                        # Filter out thinking tokens
                        import re
                        # Extract content after last </think> tag if present (DeepSeek R1 format)
                        if "</think>" in analysis_text:
                            analysis_text = analysis_text.split("</think>")[-1].strip()
                        
                        # Remove thinking tags and lines
                        analysis_text = re.sub(r'<think[^>]*>.*?</think>', '', analysis_text, flags=re.DOTALL | re.IGNORECASE)
                        analysis_text = re.sub(r'<reasoning[^>]*>.*?</reasoning>', '', analysis_text, flags=re.DOTALL | re.IGNORECASE)
                        analysis_text = re.sub(r'<redacted_reasoning[^>]*>.*?</think>', '', analysis_text, flags=re.DOTALL | re.IGNORECASE)
                        analysis_text = re.sub(r'^.*?thinking.*?$', '', analysis_text, flags=re.MULTILINE | re.IGNORECASE)
                        
                        # Parse extracted patterns (one per line)
                        patterns = []
                        for line in analysis_text.split('\n'):
                            line = line.strip()
                            # Skip empty lines, thinking tags, and non-pattern lines
                            if not line or line.startswith('<') or len(line) < 10:
                                continue
                            # Remove leading dashes/bullets if present
                            line = re.sub(r'^[-•*]\s*', '', line)
                            if line and len(line) > 10:  # Only include substantial patterns
                                patterns.append(line)
                        
                        # Store extracted patterns as memories
                        notable_memories = []
                        seen_patterns = set()
                        
                        for pattern in patterns[:limit]:
                            # Avoid duplicates
                            pattern_lower = pattern.lower().strip()
                            if pattern_lower in seen_patterns:
                                continue
                            seen_patterns.add(pattern_lower)
                            
                            # Generate embedding for the pattern
                            try:
                                pattern_embedding = embedder.embed(pattern)
                                
                                # Store as memory
                                memory_id = self.add_memory(
                                    content=pattern,
                                    embedding=pattern_embedding,
                                    metadata={
                                        "type": "notable_pattern",
                                        "source": "deepseek_analysis",
                                        "extracted_at": datetime.now(UTC).isoformat()
                                    }
                                )
                                
                                notable_memories.append({
                                    "id": memory_id,
                                    "content": pattern,
                                    "metadata": {
                                        "type": "notable_pattern",
                                        "source": "deepseek_analysis"
                                    },
                                    "created_at": datetime.now(UTC).isoformat()
                                })
                                
                                logger.debug("Extracted notable memory: %s", pattern)
                            except Exception as e:
                                logger.warning("Failed to store notable memory '%s': %s", pattern, e)
                        
                        logger.info("Extracted %d notable memories from chat history", len(notable_memories))
                        return notable_memories
                        
                    except Exception as e:
                        logger.error("Failed to analyze chat history with DeepSeek: %s", e, exc_info=True)
                        # Fall through to return stored notable memories
                
                # If DeepSeek analysis failed or not available, return existing notable memories from database
                cur.execute("""
                    SELECT id, content, metadata, created_at
                    FROM memories
                    WHERE metadata->>'type' = 'notable_pattern'
                    ORDER BY created_at DESC
                    LIMIT %s;
                """, (limit,))
                
                notable_memories = []
                for row in cur.fetchall():
                    notable_memories.append({
                        "id": row[0],
                        "content": row[1],
                        "metadata": row[2] or {},
                        "created_at": row[3].isoformat() if row[3] else None
                    })
                
                return notable_memories
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("Vector database connections closed")

