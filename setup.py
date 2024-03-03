
from setuptools import setup

import liion


setup(
    name="LiIon",
    version=liion.__version__,
    author="Evgeny P. Kurbatov",
    author_email="evgeny.p.kurbatov@gmail.com",
    packages=['liion'],
    url='https://github.com/evgenykurbatov/liion',
    description="Model for a lithium-ion battery",
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
)
