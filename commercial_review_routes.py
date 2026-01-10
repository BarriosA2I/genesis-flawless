"""
================================================================================
COMMERCIAL REVIEW API ROUTES
================================================================================
FastAPI routes for the commercial review queue and publish workflow.

Include this router in the main flawless_api.py:
    from commercial_review_routes import router as review_router
    app.include_router(review_router)

Author: Barrios A2I | Version: 1.0.0 | January 2026
================================================================================
"""

import logging
import os
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from commercial_review import (
    CommercialReviewManager, Commercial, CommercialStatus,
    create_review_manager, on_production_complete
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/commercials", tags=["Commercial_Review"])

# Global review manager (initialized on import)
_review_manager: Optional[CommercialReviewManager] = None


def get_review_manager() -> CommercialReviewManager:
    """Get or create the review manager singleton"""
    global _review_manager
    if _review_manager is None:
        _review_manager = create_review_manager()
        logger.info("CommercialReviewManager initialized")
    return _review_manager


async def initialize_review_manager(redis_client=None):
    """Initialize review manager with Redis client (called from main app startup)"""
    global _review_manager
    _review_manager = create_review_manager(redis_client=redis_client)
    logger.info("CommercialReviewManager initialized with Redis")


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class PublishRequest(BaseModel):
    """Request to publish a commercial"""
    notify_client: bool = Field(False, description="Send email notification to client")
    client_email: Optional[str] = Field(None, description="Override client email for notification")


class RejectRequest(BaseModel):
    """Request to reject a commercial"""
    reason: str = Field("", description="Rejection reason")


class CreateCommercialRequest(BaseModel):
    """Request to manually create a commercial"""
    session_id: str = Field("", description="Production session ID")
    title: str = Field("Test Commercial", description="Commercial title")
    video_url: str = Field(..., description="Video URL (required)")
    business_name: str = Field("", description="Business name")
    industry: str = Field("", description="Industry vertical")
    client_email: Optional[str] = Field(None, description="Client email for notifications")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    duration_seconds: int = Field(64, description="Video duration in seconds")


# =============================================================================
# ROUTES
# =============================================================================

@router.get("/queue")
async def get_review_queue(
    status: Optional[str] = Query(None, description="Filter by status: review, published, delivered, rejected"),
    industry: Optional[str] = Query(None, description="Filter by industry"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Get commercials in the review queue.

    By default returns all commercials. Use filters to narrow results.
    Results are sorted by creation date (newest first).

    **Status Values:**
    - `review` - Ready for internal review (default after generation)
    - `published` - Sent to video preview app
    - `delivered` - Client notified
    - `rejected` - Did not pass review
    """
    review_manager = get_review_manager()

    status_filter = None
    if status:
        try:
            status_filter = CommercialStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    commercials = await review_manager.get_review_queue(
        status=status_filter,
        industry=industry,
        limit=limit,
        offset=offset
    )

    return {
        "commercials": [c.to_dict() for c in commercials],
        "count": len(commercials),
        "filters": {
            "status": status,
            "industry": industry
        }
    }


@router.get("/stats")
async def get_queue_stats():
    """
    Get statistics about the commercial review queue.

    Returns counts by status and industry.
    """
    review_manager = get_review_manager()
    return await review_manager.get_queue_stats()


@router.get("/{commercial_id}")
async def get_commercial(commercial_id: str):
    """
    Get a specific commercial by ID.

    Returns full commercial details including all metadata.
    """
    review_manager = get_review_manager()

    commercial = await review_manager.get_commercial(commercial_id)
    if not commercial:
        raise HTTPException(status_code=404, detail="Commercial not found")

    return commercial.to_dict()


@router.post("/{commercial_id}/publish")
async def publish_commercial(commercial_id: str, request: PublishRequest):
    """
    Publish a commercial to the video preview app.

    **Workflow:**
    1. Validates commercial is in REVIEW status
    2. Sends to video-preview-theta.vercel.app via API
    3. Updates status to PUBLISHED
    4. Optionally sends email notification to client

    **Returns:**
    - `preview_url` - Gallery URL for the published commercial
    - `notification` - Email notification result (if requested)
    """
    review_manager = get_review_manager()

    # Update client email if provided
    if request.client_email:
        await review_manager.update_commercial(
            commercial_id,
            {"client_email": request.client_email}
        )

    result = await review_manager.publish_commercial(
        commercial_id,
        notify_client=request.notify_client
    )

    if not result["success"]:
        status_code = 404 if "not found" in result.get("error", "").lower() else 400
        raise HTTPException(status_code=status_code, detail=result["error"])

    return result


@router.post("/{commercial_id}/reject")
async def reject_commercial(commercial_id: str, request: RejectRequest):
    """
    Reject a commercial from the review queue.

    Marks the commercial as REJECTED with the provided reason.
    Rejected commercials remain in the system for record-keeping.
    """
    review_manager = get_review_manager()

    result = await review_manager.reject_commercial(
        commercial_id,
        reason=request.reason
    )

    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@router.post("/create")
async def create_commercial_manually(request: CreateCommercialRequest):
    """
    Manually create a commercial in the review queue.

    **Use Cases:**
    - Testing the publish workflow
    - Adding externally generated commercials
    - Re-adding rejected commercials for review

    In production, commercials are created automatically after
    RAGNAROK pipeline completes via the `on_production_complete` hook.
    """
    review_manager = get_review_manager()

    if not request.video_url:
        raise HTTPException(status_code=400, detail="video_url is required")

    commercial = await review_manager.create_commercial(
        session_id=request.session_id or str(uuid.uuid4()),
        title=request.title,
        video_url=request.video_url,
        business_name=request.business_name,
        industry=request.industry,
        client_email=request.client_email,
        thumbnail_url=request.thumbnail_url,
        duration_seconds=request.duration_seconds
    )

    return commercial.to_dict()


@router.put("/{commercial_id}")
async def update_commercial(commercial_id: str, updates: dict):
    """
    Update a commercial's fields.

    Allowed fields: title, review_notes, client_email, tags
    """
    review_manager = get_review_manager()

    # Whitelist allowed fields
    allowed_fields = {"title", "review_notes", "client_email", "tags"}
    filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}

    if not filtered_updates:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    commercial = await review_manager.update_commercial(commercial_id, filtered_updates)
    if not commercial:
        raise HTTPException(status_code=404, detail="Commercial not found")

    return commercial.to_dict()


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def include_review_routes(app):
    """
    Include review routes in a FastAPI app.

    Usage in flawless_api.py:
        from commercial_review_routes import include_review_routes
        include_review_routes(app)
    """
    app.include_router(router)
    logger.info("Commercial review routes included")
