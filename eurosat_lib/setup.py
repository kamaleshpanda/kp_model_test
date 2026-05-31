from setuptools import setup, find_packages

setup(
    name="eurosat_model",
    version="0.1.0",
    description="Multimodal EuroSAT Classification Model",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "eurosat_model": ["weights/*.pth"],
    },
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "rasterio",
        "Pillow",
        "scikit-image",
    ],
)
