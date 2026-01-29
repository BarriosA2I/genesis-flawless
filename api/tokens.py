"""
Token API Endpoints for Barrios A2I Commercial Lab
Handles token balance, additions, deductions, and transaction history.
Uses Supabase as the database backend.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Supabase client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logging.warning("supabase-py not installed. Token system unavailable.")

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", os.getenv("SUPABASE_ANON_KEY", ""))

# Initialize Supabase client
_supabase: Optional[Client] = None

def get_supabase() -> Optional[Client]:
    """Get or create Supabase client."""
    global _supabase
    if _supabase is None and SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_KEY:
        try:
            _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Supabase client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase: {e}")
    return _supabase

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class TokenBalanceResponse(BaseModel):
    user_id: str
    balance: int
    plan_type: Optional[str]
    email: Optional[str]

class AddTokensRequest(BaseModel):
    user_id: str
    amount: int
    transaction_type: str  # 'purchase', 'subscription', 'refund'
    description: Optional[str] = None
    stripe_payment_id: Optional[str] = None
    email: Optional[str] = None
    plan_type: Optional[str] = None

class DeductTokensRequest(BaseModel):
    user_id: str
    amount: int
    description: Optional[str] = None

class TokenTransaction(BaseModel):
    id: str
    user_id: str
    amount: int
    transaction_type: str
    description: Optional[str]
    stripe_payment_id: Optional[str]
    created_at: str

class TransactionsResponse(BaseModel):
    user_id: str
    transactions: List[TokenTransaction]
    total_count: int

class OperationResponse(BaseModel):
    success: bool
    message: str
    balance: Optional[int] = None

# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(prefix="/api", tags=["tokens"])

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/user/tokens", response_model=TokenBalanceResponse)
async def get_token_balance(user_id: str = Query(..., description="User ID")):
    """Get user's current token balance and plan info."""
    supabase = get_supabase()
    if not supabase:
        raise HTTPException(status_code=503, detail="Token service unavailable")

    try:
        # Query user_tokens table
        result = supabase.table("user_tokens").select("*").eq("user_id", user_id).execute()

        if result.data and len(result.data) > 0:
            user = result.data[0]
            return TokenBalanceResponse(
                user_id=user_id,
                balance=user.get("tokens_balance", 0),
                plan_type=user.get("plan_type"),
                email=user.get("email")
            )
        else:
            # User not found - return 0 balance
            return TokenBalanceResponse(
                user_id=user_id,
                balance=0,
                plan_type=None,
                email=None
            )
    except Exception as e:
        logger.error(f"Error getting token balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tokens/add", response_model=OperationResponse)
async def add_tokens(request: AddTokensRequest):
    """Add tokens to a user's balance. Called by Stripe webhook or admin."""
    supabase = get_supabase()
    if not supabase:
        raise HTTPException(status_code=503, detail="Token service unavailable")

    try:
        # Check if user exists
        existing = supabase.table("user_tokens").select("*").eq("user_id", request.user_id).execute()

        if existing.data and len(existing.data) > 0:
            # Update existing user
            current_balance = existing.data[0].get("tokens_balance", 0)
            new_balance = current_balance + request.amount

            update_data = {"tokens_balance": new_balance}
            if request.plan_type:
                update_data["plan_type"] = request.plan_type
            if request.email:
                update_data["email"] = request.email

            supabase.table("user_tokens").update(update_data).eq("user_id", request.user_id).execute()
        else:
            # Create new user record
            new_balance = request.amount
            supabase.table("user_tokens").insert({
                "user_id": request.user_id,
                "email": request.email,
                "tokens_balance": new_balance,
                "plan_type": request.plan_type
            }).execute()

        # Log transaction
        supabase.table("token_transactions").insert({
            "user_id": request.user_id,
            "amount": request.amount,
            "transaction_type": request.transaction_type,
            "description": request.description or f"Added {request.amount} tokens",
            "stripe_payment_id": request.stripe_payment_id
        }).execute()

        logger.info(f"Added {request.amount} tokens to user {request.user_id}. New balance: {new_balance}")

        return OperationResponse(
            success=True,
            message=f"Added {request.amount} tokens",
            balance=new_balance
        )
    except Exception as e:
        logger.error(f"Error adding tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tokens/deduct", response_model=OperationResponse)
async def deduct_tokens(request: DeductTokensRequest):
    """Deduct tokens from a user's balance. Called when generation starts."""
    supabase = get_supabase()
    if not supabase:
        raise HTTPException(status_code=503, detail="Token service unavailable")

    try:
        # Get current balance
        result = supabase.table("user_tokens").select("*").eq("user_id", request.user_id).execute()

        if not result.data or len(result.data) == 0:
            return OperationResponse(
                success=False,
                message="User not found",
                balance=0
            )

        current_balance = result.data[0].get("tokens_balance", 0)

        # Check if sufficient tokens
        if current_balance < request.amount:
            return OperationResponse(
                success=False,
                message=f"Insufficient tokens. Have {current_balance}, need {request.amount}",
                balance=current_balance
            )

        # Deduct tokens
        new_balance = current_balance - request.amount
        supabase.table("user_tokens").update({"tokens_balance": new_balance}).eq("user_id", request.user_id).execute()

        # Log transaction (negative amount)
        supabase.table("token_transactions").insert({
            "user_id": request.user_id,
            "amount": -request.amount,
            "transaction_type": "generation",
            "description": request.description or f"Used {request.amount} tokens for generation"
        }).execute()

        logger.info(f"Deducted {request.amount} tokens from user {request.user_id}. New balance: {new_balance}")

        return OperationResponse(
            success=True,
            message=f"Deducted {request.amount} tokens",
            balance=new_balance
        )
    except Exception as e:
        logger.error(f"Error deducting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/transactions", response_model=TransactionsResponse)
async def get_transactions(
    user_id: str = Query(..., description="User ID"),
    limit: int = Query(20, description="Max transactions to return")
):
    """Get user's token transaction history."""
    supabase = get_supabase()
    if not supabase:
        raise HTTPException(status_code=503, detail="Token service unavailable")

    try:
        result = supabase.table("token_transactions")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()

        transactions = []
        for t in result.data or []:
            transactions.append(TokenTransaction(
                id=t["id"],
                user_id=t["user_id"],
                amount=t["amount"],
                transaction_type=t["transaction_type"],
                description=t.get("description"),
                stripe_payment_id=t.get("stripe_payment_id"),
                created_at=t["created_at"]
            ))

        # Get total count
        count_result = supabase.table("token_transactions")\
            .select("*", count="exact")\
            .eq("user_id", user_id)\
            .execute()

        return TransactionsResponse(
            user_id=user_id,
            transactions=transactions,
            total_count=count_result.count or len(transactions)
        )
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HELPER FUNCTIONS (for use by other modules)
# ============================================================================

async def check_user_tokens(user_id: str) -> Dict[str, Any]:
    """Check user's token balance. Returns dict with balance and plan_type."""
    supabase = get_supabase()
    if not supabase:
        return {"balance": 0, "plan_type": None, "error": "Service unavailable"}

    try:
        result = supabase.table("user_tokens").select("*").eq("user_id", user_id).execute()
        if result.data and len(result.data) > 0:
            return {
                "balance": result.data[0].get("tokens_balance", 0),
                "plan_type": result.data[0].get("plan_type")
            }
        return {"balance": 0, "plan_type": None}
    except Exception as e:
        logger.error(f"Error checking tokens: {e}")
        return {"balance": 0, "plan_type": None, "error": str(e)}


async def use_tokens(user_id: str, amount: int, description: str = None) -> bool:
    """Attempt to use tokens. Returns True if successful, False if insufficient."""
    supabase = get_supabase()
    if not supabase:
        return False

    try:
        result = supabase.table("user_tokens").select("tokens_balance").eq("user_id", user_id).execute()
        if not result.data or result.data[0].get("tokens_balance", 0) < amount:
            return False

        new_balance = result.data[0]["tokens_balance"] - amount
        supabase.table("user_tokens").update({"tokens_balance": new_balance}).eq("user_id", user_id).execute()

        supabase.table("token_transactions").insert({
            "user_id": user_id,
            "amount": -amount,
            "transaction_type": "generation",
            "description": description or f"Used {amount} tokens"
        }).execute()

        return True
    except Exception as e:
        logger.error(f"Error using tokens: {e}")
        return False
