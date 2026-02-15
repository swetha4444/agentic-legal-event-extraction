"""
Budget check: cap number of LLM API calls per run.
"""
from typing import Optional


class BudgetExceededError(Exception):
    """Raised when max_calls is reached."""

    pass


class CallBudgetChecker:
    """Tracks API call count and enforces max_calls from config."""

    def __init__(self, max_calls: Optional[int] = None):
        self.max_calls = max_calls  # None = no limit
        self.call_count = 0

    def check(self) -> None:
        """Raise BudgetExceededError if over budget. Call before each completion()."""
        if self.max_calls is not None and self.call_count >= self.max_calls:
            raise BudgetExceededError(
                f"Budget exceeded: {self.call_count} calls (max_calls={self.max_calls})"
            )

    def record_call(self) -> None:
        """Call after each successful completion()."""
        self.call_count += 1

    @property
    def remaining(self) -> Optional[int]:
        """None if no limit, else remaining calls."""
        if self.max_calls is None:
            return None
        return max(0, self.max_calls - self.call_count)
