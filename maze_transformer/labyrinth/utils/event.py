"""Utility class for dispatching events."""

from typing import Callable, Generic, Optional, TypeVar


T = TypeVar('T')


class EventDispatcher(Generic[T]):
    """Mixin class for dispatching events when state changes."""

    def __init__(self, event_listener: Optional[Callable[[T], None]] = None) -> None:
        """Initialize an EventDispatcher with an optional event listener."""
        self.event_listener = event_listener

    def on_state_changed(self, state: T) -> None:
        """To invoke any event listeners, subclasses should call this method with their new state."""
        if self.event_listener is not None:
            try:
                self.event_listener(state)
            except Exception as e:
                print(f'Caught exception while running event listener: {e}')