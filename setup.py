import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="til-23-finals",
    version="2023.2.0",
    author="The Organizers of Today-I-Learned AI",
    author_email="",
    description="Finals Robotics challenge for TIL2023",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/til-23/til-23-finals-public",
    project_urls={
        "Bug Tracker": "https://github.com/til-23/til-23-finals-public/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        'numpy >= 1.20',
        'scipy >= 1.7.0',
        'opencv-contrib-python >= 4.5',
        'matplotlib >= 3.1.2',
        'urllib3 >= 1.25.8',
        'pyyaml >= 5.3',
        'librosa >= 0.9.1',
        'termcolor',
        'flask',
        'pynput >= 1.7.6',
        'rich >= 12.4.4',
        'Flask-Cors >= 3.0.10'
    ],
    entry_points = {
        'console_scripts': [
            'til-simulator=tilsim.simulator:main',
            'til-scoring=tilscoring.server:main',
            'til-judge=tilscoring.visualizer:main',
        ]
    },
)