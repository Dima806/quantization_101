"""quantization_101: model quantization built from scratch on CPU.

The thesis, in code: a model's weights carry far more numeric precision than its
behavior needs, so fp32 -> int8 (and often int4) is nearly free; BUT a single
outlier stretches the scale and collapses the ordinary weights, which is why
per-channel scales and outlier handling are the whole art.
"""

__version__ = "0.1.0"
