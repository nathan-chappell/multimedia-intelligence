from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from multimedia_intelligence.db import Base


class AssetRow(Base):
    """Canonical immutable upload stored in our object store."""

    __tablename__ = "assets"
    __table_args__ = (
        Index(
            "ix_assets_owner_collection_state_cursor",
            "owner_id",
            "collection_id",
            "state",
            "created_at",
            "id",
        ),
        Index(
            "ix_assets_owner_collection_state_filename",
            "owner_id",
            "collection_id",
            "state",
            "filename",
            "id",
        ),
    )
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(128), index=True)
    collection_id: Mapped[str | None] = mapped_column(
        ForeignKey("file_collections.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    filename: Mapped[str] = mapped_column(String(1024))
    media_type: Mapped[str] = mapped_column(String(255))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    bucket: Mapped[str] = mapped_column(String(255))
    object_key: Mapped[str] = mapped_column(String(2048), unique=True)
    etag: Mapped[str | None] = mapped_column(String(255), nullable=True)
    version_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    state: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class ThreadAssetIncludeRow(Base):
    """Reversible inclusion; deleting it must not delete the original asset."""

    __tablename__ = "thread_asset_includes"
    __table_args__ = (
        UniqueConstraint("thread_id", "asset_id"),
        Index(
            "ix_thread_includes_owner_thread_state_cursor",
            "owner_id",
            "thread_id",
            "state",
            "created_at",
            "id",
        ),
    )
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    thread_id: Mapped[str] = mapped_column(
        ForeignKey("chat_threads.id", ondelete="CASCADE"), index=True
    )
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="RESTRICT"), index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    user_intent: Mapped[str | None] = mapped_column(Text, nullable=True)
    intent_kind: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class DerivedArtifactRow(Base):
    """Regenerable output; provider IDs never substitute for bucket locations."""

    __tablename__ = "derived_artifacts"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    include_id: Mapped[str] = mapped_column(
        ForeignKey("thread_asset_includes.id", ondelete="CASCADE"), index=True
    )
    source_asset_id: Mapped[str] = mapped_column(
        ForeignKey("assets.id", ondelete="RESTRICT"), index=True
    )
    kind: Mapped[str] = mapped_column(String(64), index=True)
    bucket: Mapped[str | None] = mapped_column(String(255), nullable=True)
    object_key: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    provider_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    state: Mapped[str] = mapped_column(String(64), index=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class FileCollectionRow(Base):
    """User-created logical partition inside the owner's single vector store."""

    __tablename__ = "file_collections"
    __table_args__ = (
        UniqueConstraint("owner_id", "name"),
        Index("ix_file_collections_public_cursor", "is_public", "created_at", "id"),
    )
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_public: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class UserCollectionSelectionRow(Base):
    """The collection globally selected for one user's current work."""

    __tablename__ = "user_collection_selections"
    owner_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    collection_id: Mapped[str] = mapped_column(
        ForeignKey("file_collections.id", ondelete="RESTRICT"), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class UserVectorStoreRow(Base):
    """One provider vector store used as the durable search index for one user."""

    __tablename__ = "user_vector_stores"
    owner_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    provider: Mapped[str] = mapped_column(String(64), default="openai")
    vector_store_id: Mapped[str] = mapped_column(String(255), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class AssetIngestionRow(Base):
    """One resumable ingestion attempt for a canonical asset."""

    __tablename__ = "asset_ingestions"
    __table_args__ = (
        UniqueConstraint("asset_id", "version"),
        Index(
            "ix_ingestions_owner_asset_active_status_version",
            "owner_id",
            "asset_id",
            "is_active",
            "status",
            "version",
        ),
        Index("ix_ingestions_collection_status", "collection_id", "status"),
    )
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    collection_id: Mapped[str] = mapped_column(
        ForeignKey("file_collections.id", ondelete="RESTRICT"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    strategy_version: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(64), index=True)
    route: Mapped[str] = mapped_column(String(64), index=True)
    prepared_json: Mapped[str] = mapped_column(Text, default="{}")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AssetIndexArtifactRow(Base):
    """Prepared or indexed representation belonging to one ingestion attempt."""

    __tablename__ = "asset_index_artifacts"
    __table_args__ = (
        Index(
            "ix_index_artifacts_ingestion_state_cursor",
            "ingestion_id",
            "state",
            "created_at",
            "id",
        ),
        Index(
            "ix_index_artifacts_owner_asset_kind_state",
            "owner_id",
            "asset_id",
            "kind",
            "state",
        ),
    )
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    ingestion_id: Mapped[str] = mapped_column(
        ForeignKey("asset_ingestions.id", ondelete="CASCADE"), index=True
    )
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    state: Mapped[str] = mapped_column(String(64), index=True)
    bucket: Mapped[str | None] = mapped_column(String(255), nullable=True)
    object_key: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    media_type: Mapped[str] = mapped_column(String(255))
    provider_file_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    provider_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    provider_checked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    provider_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
