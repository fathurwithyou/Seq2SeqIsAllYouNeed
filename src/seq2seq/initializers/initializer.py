class Initializer:
    """Initializer base class: all initializers inherit from this class."""

    def __call__(self, shape, dtype=None):
        """Returns an array initialized as specified by the initializer."""
        raise NotImplementedError(
            "Initializer subclasses must implement the `__call__()` method."
        )

    def get_config(self):
        """Returns the initializer's JSON-serializable configuration."""
        return {}

    @classmethod
    def from_config(cls, config):
        """Instantiates an initializer from a configuration dictionary."""
        return cls(**config)

    def clone(self):
        return self.__class__.from_config(self.get_config())


__all__ = ["Initializer"]
