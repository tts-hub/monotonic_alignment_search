# SPDX-FileCopyrightText: Enno Hermann
#
# SPDX-License-Identifier: MIT

import numpy as np
import numpy.typing as npt

def maximum_path_c(
    paths: npt.NDArray[np.int32],
    values: npt.NDArray[np.float32],
    t_xs: npt.NDArray[np.int32],
    t_ys: npt.NDArray[np.int32],
    max_neg_val: float = ...,
) -> None: ...
