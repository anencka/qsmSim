"""qsmsim - numerical QSM simulation and reconstruction comparison.

phantom -> dipole forward model -> multi-echo GRE (noise, transceive phase,
shim) -> complex field fit -> background removal (SHARP / V-SHARP / PDF) ->
dipole inversion (TKD / CF-L2 / TV-ADMM / MEDI-like) -> metrics + figures.
"""
__version__ = "0.1.0"
