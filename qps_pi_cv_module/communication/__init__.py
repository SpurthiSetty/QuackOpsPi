"""
Communication module for backend server interaction.
"""

from .backend_client import BackendClient, HttpBackendClient, MessageType

__all__ = [
    "BackendClient",
    "HttpBackendClient",
    "MessageType",
]
