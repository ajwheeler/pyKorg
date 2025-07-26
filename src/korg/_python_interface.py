"""
This module defines the python interface to Korg
"""

import os
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, ParamSpec
from ._julia_import import jl, Korg

from juliacall import VectorValue as jlVectorValue

# this is for typing purposes (they aren't necessary starting in python 3.12)
T = TypeVar("T")
P = ParamSpec("P")


def _perfect_jl_shadowing(fn: Callable[P, T]) -> Callable[P, T]:
    """A decorator for functions that perfectly shadows a Korg

    The main purpose of this function is to add the Korg docstring. This should
    be used somewhat sparingly (i.e. in cases when we are confident that the
    docstring will never mention Julia-specific types)
    """
    _recycle_jl_docstring(fn)
    return fn


def _recycle_jl_docstring(fn: Callable):
    # this is experimental (to be used sparingly in cases when we are confident that
    # the docstrings won't mention Julia specific types)
    #
    # this is separate from _perfect_jl_shadowing because there may be a lot of heavy
    # lifting (and there could conceivably be cases where we want to reuse part of a
    # docstring)

    # TODO: we need to figure out how to best translate Documenter.jl flavored markdown
    #       to restructured text. Since the docstrings of all public Korg functions
    #       largely share a common structure, it probably wouldn't be bad to do this
    #       with a few regex statements
    jl_docstring = jl.seval(f"(@doc Korg.{fn.__name__}).text[1]")

    if jl_docstring.startswith(f"    {fn.__name__}("):
        first_newline = jl_docstring.index("\n")
        fn.__doc__ = jl_docstring[first_newline:].lstrip()
    else:
        raise RuntimeError(
            f"There was a problem getting the docstring for {fn.__name__}"
        )


class LineList:
    """A lightweight class that wraps a line list.

    You shouldn't try to initialize this class directly. Instead, you should rely upon
    functions like :py:func:`~korg.get_APOGEE_DR17_linelist`,
    :py:func:`~korg.get_GES_linelist`, etc.
    """

    _lines: jlVectorValue

    def __init__(self, lines: jlVectorValue):
        self._lines = lines

    def __len__(self) -> int:
        return len(self._lines)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        length = len(self)
        return f"{name}(<{length} lines>)"


@_perfect_jl_shadowing
def get_APOGEE_DR17_linelist(*, include_water: bool = True) -> LineList:
    return LineList(Korg.get_APOGEE_DR17_linelist(include_water=include_water))


@_perfect_jl_shadowing
def get_GALAH_DR3_linelist() -> LineList:
    return LineList(Korg.get_GALAH_DR3_linelist())


@_perfect_jl_shadowing
def get_GES_linelist(*, include_molecules: bool = True) -> LineList:
    return LineList(Korg.get_GES_linelist(include_molecules=include_molecules))


@_perfect_jl_shadowing
def get_VALD_solar_linelist() -> LineList:
    return LineList(Korg.get_VALD_solar_linelist())


# we can't currently reuse the exact Julia signature since the Julia signature
# explicitly states that it returns a vector of lines
def read_linelist(
    fname: os.PathLike,
    *,
    format: str | None = None,
    isotopic_abundances: Mapping[int, Mapping[float, float]] | None = None,
) -> LineList:
    # coerce fname to a string
    coerced_fname = os.fsdecode(fname)

    # build up kwargs (we have to play some games here since we can't natively
    # represent the default values in python)
    kwargs: dict[str, Any] = {}
    if format is not None:
        kwargs["format"] = format
    if isotopic_abundances is not None:
        kwargs["isotopic_abundances"] = isotopic_abundances
    return LineList(Korg.read_linelist(coerced_fname, **kwargs))
