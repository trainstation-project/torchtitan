# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for the perplexity (bits per byte) tasks."""


def doc_to_text(doc: dict) -> str:
    return str(doc["text"])
