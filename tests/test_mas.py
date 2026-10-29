# SPDX-FileCopyrightText: Enno Hermann
#
# SPDX-License-Identifier: MIT

"""Tests for monotonic alignment search."""

import pytest
import torch

from monotonic_alignment_search import maximum_path


def test_mas() -> None:
    """Basic functionality test."""
    for shape in [(1, 20, 40), (10, 20, 40)]:
        value = torch.rand(shape)
        mask = torch.ones_like(value)
        assert torch.all(
            maximum_path(value, mask, "cython") == maximum_path(value, mask, "numpy"),
        )

    value = torch.rand((1, 20, 40))
    mask = torch.ones_like(value)
    with pytest.raises(NotImplementedError):
        maximum_path(value, mask, "INVALID")  # ty: ignore[invalid-argument-type]
