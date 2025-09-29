from setuptools import setup, find_packages

setup(
    name="conversational-ai",
    version="0.1.0",
    description="Conversational AI module for RAVANA AGI system",
    author="RAVANA Development Team",
    packages=find_packages(),
    install_requires=[
        "discord.py>=2.0.0",
        "python-telegram-bot>=20.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "conversational-ai=conversational_ai.main:main",
        ],
    },
)
