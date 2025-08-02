"""Placeholder bimanual robot interface."""


class Bimanual:
    """Minimal interface for the real robot.

    The methods here do not perform any hardware communication but allow
    import statements to succeed in environments without the robot stack.
    """

    def __init__(self) -> None:
        self.connected = False

    def connect(self) -> None:
        """Simulate establishing a connection to the robot."""
        self.connected = True

    def disconnect(self) -> None:
        """Simulate terminating the connection."""
        self.connected = False
