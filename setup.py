from setuptools import setup, find_packages

setup(
    name="yt2epub",
    version="1.0.0",
    author="Mahmud Esad Yazar",
    description="Convert YouTube videos into EPUBs with AI transcription, formatting, and cover generation.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "yt_dlp",
        "faster-whisper",
        "google-generativeai",
        "diffusers",
        "transformers",
        "torch",
        "ebooklib",
        "Pillow",
        "tk",
        "tkinterdnd2",
        "llama-cpp-python",
        "safetensors",
        "huggingface-hub",
        "python-dotenv"
    ],
    entry_points={
        "console_scripts": [
            "yt2epub=yt2epub.main:launch"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires=">=3.8",
)