from distutils.core import setup

setup(
    name="interlocking_backprop",
    version="0.1dev",
    packages=["interlocking_backprop"],
    install_requires=["torch", "torchvision", "numpy"],
)
