from setuptools import Extension, setup

module = Extension(
    "symnmf_c",
    sources=["symnmfmodule.c", "symnmf.c"]       
)

setup(
    name="symnmf_c",
    version="1.0",
    description="A Python wrapper for the Symmetric NMF algorithm",
    ext_modules=[module]
)
