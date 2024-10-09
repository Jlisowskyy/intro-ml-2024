"""
Module containing the model
"""
from pydantic import BaseModel


class ModelResponse(BaseModel):
    """Class representing the output of the model

    Attributes
    ----------
    response: :class:`str`
        Model's response.
    """
    response: str
