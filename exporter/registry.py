"""Model registry — register and look up model definitions."""

from exporter.base import BaseModel

_models: dict[str, BaseModel] = {}


def register(model: BaseModel) -> None:
    """Register a model definition."""
    _models[model.name] = model


def get(name: str) -> BaseModel | None:
    """Get a model by name."""
    return _models.get(name)


def all_models() -> list[BaseModel]:
    """Return all registered models."""
    return list(_models.values())


def names() -> list[str]:
    """Return all registered model names."""
    return list(_models.keys())
