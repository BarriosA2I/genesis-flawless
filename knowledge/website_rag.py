"""
Website Knowledge RAG System
Automatically indexes barriosa2i.com content for real-time business knowledge retrieval.

This prevents AI knowledge drift by always having up-to-date website content available.
"""

import os
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Any

import httpx
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)


class WebsiteKnowledgeRAG:
    """
    RAG system that learns from barriosa2i.com website content.

    Architecture:
    Website -> Firecrawl -> Chunks -> OpenAI Embeddings -> Qdrant Vector DB
                                                                ↓
    User Question -> Embed -> Similarity Search -> Context -> Claude
    """

    COLLECTION_NAME = "barrios-website-knowledge"
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSION = 1536
    CHUNK_SIZE = 500  # tokens
    CHUNK_OVERLAP = 50  # tokens overlap between chunks

    # Pages to crawl from barriosa2i.com
    TARGET_URLS = [
        "https://www.barriosa2i.com",
        "https://www.barriosa2i.com/creative-director",
        "https://www.barriosa2i.com/pricing",
        "https://www.barriosa2i.com/nexus-personal",
        "https://www.barriosa2i.com/founder",
        "https://www.barriosa2i.com/status",
    ]

    def __init__(self):
        """Initialize connections to Qdrant, OpenAI, and Firecrawl."""
        # Qdrant for vector storage
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url and qdrant_api_key:
            self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            logger.warning("Qdrant credentials not found - using in-memory storage")
            self.qdrant = QdrantClient(":memory:")

        # OpenAI for embeddings
        self.openai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Firecrawl for web scraping
        self.firecrawl_key = os.getenv("FIRECRAWL_API_KEY")

        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.qdrant.get_collections().collections
            if not any(c.name == self.COLLECTION_NAME for c in collections):
                self.qdrant.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")

    async def crawl_website(self) -> List[Dict]:
        """
        Crawl barriosa2i.com using Firecrawl API.

        Returns:
            List of page data with URL, content, and metadata
        """
        if not self.firecrawl_key:
            logger.warning("Firecrawl API key not set - using fallback scraper")
            return await self._fallback_scrape()

        pages = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            for url in self.TARGET_URLS:
                try:
                    response = await client.post(
                        "https://api.firecrawl.dev/v0/scrape",
                        headers={"Authorization": f"Bearer {self.firecrawl_key}"},
                        json={
                            "url": url,
                            "pageOptions": {
                                "onlyMainContent": True,
                                "includeHtml": False
                            }
                        }
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if data.get("success"):
                            pages.append({
                                "url": url,
                                "content": data.get("data", {}).get("markdown", ""),
                                "title": data.get("data", {}).get("title", ""),
                                "crawled_at": datetime.utcnow().isoformat()
                            })
                            logger.info(f"Crawled: {url}")
                    else:
                        logger.warning(f"Failed to crawl {url}: {response.status_code}")

                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}")

        return pages

    async def _fallback_scrape(self) -> List[Dict]:
        """
        Fallback scraper using httpx when Firecrawl is not available.
        Uses BeautifulSoup-like extraction from raw HTML.
        """
        pages = []

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        async with httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True) as client:
            for url in self.TARGET_URLS:
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        # Basic content extraction (strip HTML tags)
                        content = self._extract_text_from_html(response.text)
                        pages.append({
                            "url": url,
                            "content": content,
                            "title": url.split("/")[-1] or "home",
                            "crawled_at": datetime.utcnow().isoformat()
                        })
                        logger.info(f"Fallback scraped: {url}")
                except Exception as e:
                    logger.error(f"Error in fallback scrape {url}: {e}")

        return pages

    def _extract_text_from_html(self, html: str) -> str:
        """Basic HTML text extraction."""
        import re
        # Remove script and style elements
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings using OpenAI text-embedding-3-small.

        Args:
            text: Text to embed

        Returns:
            1536-dimensional embedding vector
        """
        try:
            response = self.openai.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [0.0] * self.EMBEDDING_DIMENSION

    def _chunk_content(self, content: str, url: str, title: str = "") -> List[Dict]:
        """
        Split content into overlapping chunks for better retrieval.

        Args:
            content: Full page content
            url: Source URL
            title: Page title

        Returns:
            List of chunk dictionaries
        """
        words = content.split()
        chunks = []

        for i in range(0, len(words), self.CHUNK_SIZE - self.CHUNK_OVERLAP):
            chunk_words = words[i:i + self.CHUNK_SIZE]
            chunk_text = " ".join(chunk_words)

            # Generate unique ID for chunk
            chunk_id = hashlib.md5(f"{url}:{i}:{chunk_text[:50]}".encode()).hexdigest()

            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "url": url,
                "title": title,
                "chunk_index": i // (self.CHUNK_SIZE - self.CHUNK_OVERLAP)
            })

        return chunks

    async def index_website(self) -> Dict[str, Any]:
        """
        Full re-index of website content.

        Returns:
            Index statistics
        """
        logger.info("Starting website re-index...")

        # 1. Crawl website
        pages = await self.crawl_website()

        if not pages:
            return {"status": "error", "message": "No pages crawled"}

        # 2. Recreate collection for clean slate
        try:
            self.qdrant.delete_collection(self.COLLECTION_NAME)
        except:
            pass

        self.qdrant.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=self.EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )

        # 3. Process and store each page
        points = []
        total_chunks = 0

        for page in pages:
            content = page.get("content", "")
            if not content or len(content) < 50:
                continue

            chunks = self._chunk_content(
                content,
                page.get("url", ""),
                page.get("title", "")
            )

            for chunk in chunks:
                embedding = self.embed_text(chunk["text"])

                # Use hash as integer ID
                point_id = int(hashlib.md5(chunk["id"].encode()).hexdigest()[:15], 16)

                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "url": chunk["url"],
                        "title": chunk["title"],
                        "chunk_index": chunk["chunk_index"],
                        "indexed_at": datetime.utcnow().isoformat()
                    }
                ))
                total_chunks += 1

        # 4. Upsert to Qdrant in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.qdrant.upsert(collection_name=self.COLLECTION_NAME, points=batch)

        logger.info(f"Indexed {total_chunks} chunks from {len(pages)} pages")

        return {
            "status": "success",
            "pages_crawled": len(pages),
            "chunks_indexed": total_chunks,
            "collection": self.COLLECTION_NAME,
            "indexed_at": datetime.utcnow().isoformat()
        }

    def query(self, question: str, top_k: int = 5) -> str:
        """
        Query website knowledge for relevant context.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve

        Returns:
            Formatted context string from relevant chunks
        """
        try:
            embedding = self.embed_text(question)

            results = self.qdrant.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=embedding,
                limit=top_k
            )

            if not results:
                return ""

            context_parts = []
            seen_urls = set()

            for result in results:
                url = result.payload.get("url", "")
                text = result.payload.get("text", "")
                score = result.score

                # Skip low confidence results
                if score < 0.5:
                    continue

                # Dedupe by URL
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                context_parts.append(f"[Source: {url}]\n{text}")

            if not context_parts:
                return ""

            return "\n\n---\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Query error: {e}")
            return ""

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.qdrant.get_collection(self.COLLECTION_NAME)
            # Handle different Qdrant client versions
            stats = {
                "collection": self.COLLECTION_NAME,
                "status": str(info.status) if hasattr(info, 'status') else "unknown"
            }
            # Try different attribute names for vector count
            if hasattr(info, 'vectors_count'):
                stats["vectors_count"] = info.vectors_count
            elif hasattr(info, 'indexed_vectors_count'):
                stats["vectors_count"] = info.indexed_vectors_count
            # Points count
            if hasattr(info, 'points_count'):
                stats["points_count"] = info.points_count
            return stats
        except Exception as e:
            return {"error": str(e)}


# Singleton instance for reuse
_rag_instance: Optional[WebsiteKnowledgeRAG] = None


def get_website_rag() -> WebsiteKnowledgeRAG:
    """Get or create singleton RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = WebsiteKnowledgeRAG()
    return _rag_instance
