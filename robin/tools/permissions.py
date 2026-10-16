import hashlib
import json
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


POLICY: Dict[str, str] = {
    "open_app": "allow",
    "project_summary": "allow",
    "code_quality": "allow",
    "file_view": "allow",
    "index_project": "allow",
    "search_project": "allow",
    "generate_image": "allow",
    "read_file": "confirm",
    "write_file": "confirm",
    "run_command": "confirm",
    "install_package": "confirm",
    "send_message": "confirm",
    "delete_project_index": "confirm",
    "delete_file": "deny",
    "permanent_delete": "deny",
}


@dataclass
class ApprovalTicket:
    """A scoped, time-bounded ticket authorizing a specific sensitive action."""
    ticket_id: str
    action: str
    target: str
    params_hash: str
    created_at: float
    expires_at: float
    approved: bool = False
    used: bool = False

    def is_expired(self, now: Optional[float] = None) -> bool:
        current = now if now is not None else time.time()
        return current > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "action": self.action,
            "target": self.target,
            "params_hash": self.params_hash,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "approved": self.approved,
            "used": self.used,
        }


def hash_parameters(params: Optional[Dict[str, Any]]) -> str:
    """Generate a deterministic SHA-256 fingerprint for capability parameters."""
    if not params:
        return ""
    try:
        normalized = json.dumps(params, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    except Exception:
        return ""


class ApprovalManager:
    """Manages short-lived cryptographic approval tickets for sensitive operations."""

    def __init__(self) -> None:
        self._tickets: Dict[str, ApprovalTicket] = {}
        self._lock = threading.Lock()

    def create_ticket(
        self,
        action: str,
        target: str = "",
        params: Optional[Dict[str, Any]] = None,
        ttl_seconds: float = 300.0,
    ) -> ApprovalTicket:
        """Create a new pending approval ticket scoped to action, target, and parameter fingerprint."""
        now = time.time()
        ticket_id = secrets.token_urlsafe(24)
        params_hash = hash_parameters(params)

        ticket = ApprovalTicket(
            ticket_id=ticket_id,
            action=action.strip().lower(),
            target=str(target).strip(),
            params_hash=params_hash,
            created_at=now,
            expires_at=now + max(0.001, ttl_seconds),
            approved=False,
            used=False,
        )

        with self._lock:
            # Prune expired tickets
            self._prune_expired_locked(now)
            self._tickets[ticket_id] = ticket

        return ticket

    def get_ticket(self, ticket_id: str) -> Optional[ApprovalTicket]:
        """Retrieve a ticket by ID if it has not expired."""
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if ticket and not ticket.is_expired():
                return ticket
            return None

    def approve_ticket(self, ticket_id: str) -> bool:
        """Mark a pending ticket as explicitly approved by the user."""
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if not ticket or ticket.is_expired() or ticket.used:
                return False
            ticket.approved = True
            return True

    def validate_and_consume(
        self,
        ticket_id: str,
        action: str,
        target: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Validate that an approval ticket matches the requested action and consume it (single-use).

        Returns:
            (is_valid, reason)
        """
        now = time.time()
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if not ticket:
                return False, "Approval ticket does not exist."
            if ticket.is_expired(now):
                return False, "Approval ticket has expired."
            if ticket.used:
                return False, "Approval ticket has already been used."
            if not ticket.approved:
                return False, "Approval ticket was not approved by the user."

            expected_action = action.strip().lower()
            if ticket.action != expected_action:
                return False, f"Ticket action mismatch: issued for '{ticket.action}', not '{expected_action}'."

            # Verify target if specified on ticket
            if ticket.target and ticket.target != str(target).strip():
                return False, f"Ticket target mismatch: issued for '{ticket.target}', not '{target}'."

            # Verify parameters hash if present
            expected_hash = hash_parameters(params)
            if ticket.params_hash and ticket.params_hash != expected_hash:
                return False, "Ticket parameter fingerprint mismatch."

            # Single-use consumption
            ticket.used = True
            return True, "Ticket successfully validated and consumed."

    def _prune_expired_locked(self, now: float) -> None:
        """Remove stale tickets (called while self._lock is held)."""
        expired = [tid for tid, t in self._tickets.items() if t.is_expired(now)]
        for tid in expired:
            del self._tickets[tid]


_default_approval_manager: Optional[ApprovalManager] = None


def get_default_approval_manager() -> ApprovalManager:
    """Return the global ApprovalManager instance."""
    global _default_approval_manager
    if _default_approval_manager is None:
        _default_approval_manager = ApprovalManager()
    return _default_approval_manager


def get_permission_policy(action: str) -> str:
    """Return the policy decision ('allow', 'confirm', or 'deny') for an action."""
    return POLICY.get(action.strip().lower(), "deny")


def can_execute(
    action: str,
    target: str = "",
    confirmed: bool = False,
    approval_token: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    strict_tickets: bool = False,
) -> bool:
    """
    Programmatic permission check.

    Returns True if:
    - Action policy is 'allow'.
    - Action policy is 'confirm' and a verified approval ticket is consumed.
    - If strict_tickets is False (legacy test mode), confirmed is True. Bare booleans
      are rejected whenever strict_tickets is True.
    """
    decision = get_permission_policy(action)
    if decision == "allow":
        return True
    if decision == "deny":
        return False

    # Action is "confirm"
    if approval_token:
        valid, _ = get_default_approval_manager().validate_and_consume(
            ticket_id=approval_token,
            action=action,
            target=target,
            params=params,
        )
        return valid

    if strict_tickets:
        return False

    return confirmed is True


def authorize(
    action: str,
    target: str = "",
    input_fn: Callable[[str], str] = input,
) -> bool:
    """Apply a policy independent of the language model with interactive prompt fallback."""
    decision = get_permission_policy(action)

    if decision == "allow":
        return True

    if decision == "deny":
        return False

    response = input_fn(
        f"\nRobin requests permission to perform '{action}' "
        f"on '{target}'.\n"
        "Type YES to approve this action: "
    )

    return response.strip() == "YES"
