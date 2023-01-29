from setuptools import find_packages
from setuptools import setup


setup(
    name='EDM_tools',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/yairdaon/tools',
    license='MIT',
    author='Yair Daon',
    author_email='yair.daon@gmail.com',
    description='EDM tools I use.'
)
