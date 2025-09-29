#!/usr/bin/env python3
"""
Database setup and migration utilities for RAVANA multi-modal memory system.
Handles PostgreSQL database creation, schema migration, and data migration from ChromaDB.
"""

import os
import sys
import logging
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
import uuid

try:
    import asyncpg
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"Required dependencies not available: {e}")
    print("Install with: pip install asyncpg chromadb")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """
    Handles database setup and migration for the multi-modal memory system.
    """

    def __init__(self, database_url: str, chroma_db_path: str = "chroma_db"):
        """
        Initialize the database migrator.

        Args:
            database_url: PostgreSQL connection URL
            chroma_db_path: Path to ChromaDB storage
        """
        self.database_url = database_url
        self.chroma_db_path = chroma_db_path
        self.conn = None

        # Load schema
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                self.schema_sql = f.read()
        else:
            logger.error(f"Schema file not found: {schema_path}")
            self.schema_sql = None

    async def connect(self):
        """Connect to PostgreSQL database."""
        try:
            self.conn = await asyncpg.connect(self.database_url)
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    async def close(self):
        """Close database connection."""
        if self.conn:
            await self.conn.close()
            logger.info("Database connection closed")

    async def check_database_exists(self) -> bool:
        """Check if the database exists and is accessible."""
        try:
            await self.connect()
            result = await self.conn.fetchval("SELECT version()")
            logger.info(f"Database accessible: {result}")
            return True
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            return False
        finally:
            await self.close()

    async def check_pgvector_extension(self) -> bool:
        """Check if pgvector extension is available."""
        try:
            await self.connect()
            result = await self.conn.fetchval(
                "SELECT COUNT(*) FROM pg_available_extensions WHERE name = 'vector'"
            )
            available = result > 0
            logger.info(f"pgvector extension available: {available}")

            if available:
                # Check if already installed
                installed = await self.conn.fetchval(
                    "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
                )
                logger.info(f"pgvector extension installed: {installed > 0}")

            return available
        except Exception as e:
            logger.error(f"pgvector check failed: {e}")
            return False
        finally:
            await self.close()

    async def create_schema(self, force: bool = False):
        """Create database schema from schema.sql."""
        if not self.schema_sql:
            logger.error("Schema SQL not available")
            return False

        try:
            await self.connect()

            # Check if tables already exist
            existing_tables = await self.conn.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
            existing_table_names = [row['table_name']
                                    for row in existing_tables]

            if existing_table_names and not force:
                logger.warning(f"Tables already exist: {existing_table_names}")
                logger.warning("Use --force to recreate schema")
                return False

            if force and existing_table_names:
                logger.info("Dropping existing tables...")
                for table_name in existing_table_names:
                    if table_name.startswith('memory_') or table_name.endswith('_memories'):
                        await self.conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")

            # Execute schema
            logger.info("Creating database schema...")
            await self.conn.execute(self.schema_sql)
            logger.info("Database schema created successfully")

            # Verify tables were created
            new_tables = await self.conn.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
            table_names = [row['table_name'] for row in new_tables]
            logger.info(f"Created tables: {table_names}")

            return True

        except Exception as e:
            logger.error(f"Schema creation failed: {e}")
            return False
        finally:
            await self.close()

    async def migrate_from_chromadb(self, batch_size: int = 100):
        """Migrate existing memories from ChromaDB to PostgreSQL."""
        try:
            # Connect to ChromaDB
            logger.info(f"Connecting to ChromaDB at: {self.chroma_db_path}")
            chroma_client = chromadb.Client(
                Settings(persist_directory=self.chroma_db_path,
                         is_persistent=True)
            )

            # Get the memories collection
            try:
                collection = chroma_client.get_collection(name='memories')
                logger.info(f"Found ChromaDB collection: memories")
            except Exception:
                logger.warning(
                    "ChromaDB collection 'memories' not found, skipping migration")
                return 0

            # Get all memories
            count = collection.count()
            logger.info(f"Found {count} memories in ChromaDB")

            if count == 0:
                return 0

            # Connect to PostgreSQL
            await self.connect()

            migrated_count = 0
            offset = 0

            while offset < count:
                # Get batch of memories
                limit = min(batch_size, count - offset)

                result = collection.get(
                    limit=limit,
                    offset=offset,
                    include=["metadatas", "documents", "embeddings"]
                )

                if not result or not result['ids']:
                    break

                # Process batch
                for i, memory_id in enumerate(result['ids']):
                    try:
                        # Extract data
                        metadata = result['metadatas'][i] if result['metadatas'] else {
                        }
                        document = result['documents'][i] if result['documents'] else ""
                        embedding = result['embeddings'][i] if result['embeddings'] else None

                        # Create memory record
                        memory_record = {
                            'id': uuid.UUID(memory_id) if self._is_valid_uuid(memory_id) else uuid.uuid4(),
                            'content_type': 'text',
                            'content_text': document or metadata.get('text', ''),
                            'content_metadata': json.dumps({
                                'migrated_from_chromadb': True,
                                'original_id': memory_id,
                                'migration_date': datetime.utcnow().isoformat()
                            }),
                            'text_embedding': embedding,
                            'created_at': self._parse_datetime(metadata.get('created_at')),
                            'last_accessed': self._parse_datetime(metadata.get('last_accessed')),
                            'access_count': metadata.get('access_count', 0),
                            'memory_type': metadata.get('type', 'episodic'),
                            'confidence_score': 1.0
                        }

                        # Insert into PostgreSQL
                        await self._insert_memory_record(memory_record)
                        migrated_count += 1

                    except Exception as e:
                        logger.error(
                            f"Failed to migrate memory {memory_id}: {e}")

                offset += limit
                logger.info(f"Migrated {migrated_count}/{count} memories...")

            logger.info(
                f"Migration completed. Migrated {migrated_count} memories")
            return migrated_count

        except Exception as e:
            logger.error(f"ChromaDB migration failed: {e}")
            return 0
        finally:
            await self.close()

    async def _insert_memory_record(self, record: Dict[str, Any]):
        """Insert a memory record into PostgreSQL."""
        query = """
            INSERT INTO memory_records (
                id, content_type, content_text, content_metadata,
                text_embedding, created_at, last_accessed, access_count,
                memory_type, confidence_score
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO NOTHING
        """

        await self.conn.execute(
            query,
            record['id'],
            record['content_type'],
            record['content_text'],
            record['content_metadata'],
            record['text_embedding'],
            record['created_at'],
            record['last_accessed'],
            record['access_count'],
            record['memory_type'],
            record['confidence_score']
        )

    def _is_valid_uuid(self, uuid_string: str) -> bool:
        """Check if string is a valid UUID."""
        try:
            uuid.UUID(uuid_string)
            return True
        except (ValueError, TypeError):
            return False

    def _parse_datetime(self, dt_string: Optional[str]) -> datetime:
        """Parse datetime string or return current time."""
        if not dt_string:
            return datetime.utcnow()

        try:
            # Try ISO format first
            return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return datetime.utcnow()

    async def verify_migration(self) -> Dict[str, Any]:
        """Verify the migration by checking record counts and data integrity."""
        try:
            await self.connect()

            # Count records
            total_count = await self.conn.fetchval("SELECT COUNT(*) FROM memory_records")

            # Count by content type
            type_counts = await self.conn.fetch(
                "SELECT content_type, COUNT(*) as count FROM memory_records GROUP BY content_type"
            )

            # Count records with embeddings
            embedding_count = await self.conn.fetchval(
                "SELECT COUNT(*) FROM memory_records WHERE text_embedding IS NOT NULL"
            )

            # Check for migrated records
            migrated_count = await self.conn.fetchval(
                "SELECT COUNT(*) FROM memory_records WHERE content_metadata::text LIKE '%migrated_from_chromadb%'"
            )

            results = {
                'total_records': total_count,
                'records_with_embeddings': embedding_count,
                'migrated_records': migrated_count,
                'content_type_breakdown': {row['content_type']: row['count'] for row in type_counts}
            }

            logger.info(f"Migration verification: {results}")
            return results

        except Exception as e:
            logger.error(f"Migration verification failed: {e}")
            return {}
        finally:
            await self.close()

    async def backup_chromadb(self, backup_path: str):
        """Create a backup of ChromaDB data."""
        try:
            import shutil

            source_path = Path(self.chroma_db_path)
            backup_path = Path(backup_path)

            if not source_path.exists():
                logger.warning(f"ChromaDB path does not exist: {source_path}")
                return False

            backup_path.parent.mkdir(parents=True, exist_ok=True)

            if backup_path.exists():
                shutil.rmtree(backup_path)

            shutil.copytree(source_path, backup_path)
            logger.info(f"ChromaDB backup created: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"ChromaDB backup failed: {e}")
            return False


def get_database_url() -> str:
    """Get database URL from environment or use default."""
    return os.getenv(
        "POSTGRES_URL",
        "postgresql://postgres:password@localhost:5432/ravana_memory"
    )


async def main():
    """Main migration script."""
    parser = argparse.ArgumentParser(
        description="RAVANA Multi-Modal Memory Database Setup")
    parser.add_argument("--database-url", default=get_database_url(),
                        help="PostgreSQL database URL")
    parser.add_argument("--chroma-path", default="chroma_db",
                        help="Path to ChromaDB storage")
    parser.add_argument("--force", action="store_true",
                        help="Force recreation of schema")
    parser.add_argument("--skip-migration", action="store_true",
                        help="Skip ChromaDB migration")
    parser.add_argument("--backup-chromadb",
                        help="Backup ChromaDB to specified path before migration")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing migration")

    args = parser.parse_args()

    migrator = DatabaseMigrator(args.database_url, args.chroma_path)

    try:
        # Check database connectivity
        logger.info("Checking database connectivity...")
        if not await migrator.check_database_exists():
            logger.error(
                "Cannot connect to database. Please check your connection settings.")
            return 1

        # Check pgvector extension
        logger.info("Checking pgvector extension...")
        if not await migrator.check_pgvector_extension():
            logger.error(
                "pgvector extension not available. Please install it first.")
            logger.info(
                "Installation guide: https://github.com/pgvector/pgvector")
            return 1

        if args.verify_only:
            # Only verify existing migration
            await migrator.verify_migration()
            return 0

        # Backup ChromaDB if requested
        if args.backup_chromadb:
            logger.info("Creating ChromaDB backup...")
            await migrator.backup_chromadb(args.backup_chromadb)

        # Create schema
        logger.info("Setting up database schema...")
        if not await migrator.create_schema(force=args.force):
            logger.error("Schema creation failed")
            return 1

        # Migrate from ChromaDB
        if not args.skip_migration:
            logger.info("Migrating data from ChromaDB...")
            migrated_count = await migrator.migrate_from_chromadb()
            logger.info(f"Migrated {migrated_count} records")

        # Verify migration
        logger.info("Verifying migration...")
        verification_results = await migrator.verify_migration()

        logger.info("Migration completed successfully!")
        logger.info("\nNext steps:")
        logger.info(
            "1. Set POSTGRES_URL environment variable if not already set")
        logger.info(
            "2. Install multi-modal dependencies: pip install -r requirements.txt")
        logger.info("3. Start the memory service: python memory.py")

        return 0

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
