"""Task heads for cancer-specific classification and regression.

Survival heads deliberately live in ``kalecancer.survival`` instead, not
here: that package is quarantined (imports only ``torch``, ``numpy``, and
``pykale``) so it can migrate cleanly into PyKale core later, a
constraint this package is not held to.
"""
