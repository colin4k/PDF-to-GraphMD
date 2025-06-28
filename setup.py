"""
Setup script for PDF-to-GraphMD
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    with open(requirements_path, encoding='utf-8') as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('http'):
                # Handle package names with version specifiers
                if '>=' in line:
                    requirements.append(line)
                elif '@' in line:
                    # Skip direct URL dependencies for now
                    continue
                else:
                    requirements.append(line)
        return requirements

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="pdf-to-graphmd",
    version="1.0.0",
    description="Automated Knowledge Graph Construction System for PDF Documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="PDF-to-GraphMD Team",
    author_email="",
    url="https://github.com/your-username/pdf-to-graphmd",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.971"
        ],
        "gpu": [
            "torch>=1.9.0",
            "torchvision>=0.10.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "pdf-to-graphmd=pdf_to_graphmd.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Researchers",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Markup",
        "Topic :: Utilities",
    ],
    keywords="pdf knowledge-graph obsidian nlp llm document-processing",
    project_urls={
        "Bug Reports": "https://github.com/your-username/pdf-to-graphmd/issues",
        "Source": "https://github.com/your-username/pdf-to-graphmd",
        "Documentation": "https://github.com/your-username/pdf-to-graphmd/wiki",
    },
    include_package_data=True,
    zip_safe=False,
)