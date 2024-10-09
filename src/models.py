"""
Module containing the model
"""
from pydantic import BaseModel


# remove the pylint warning later
class ModelResponse(BaseModel):  # pylint: disable=too-few-public-methods
    """Class representing the output of the model

    Attributes
    ----------
    response: :class:`str`
        Model's response.
    """
    response: str
