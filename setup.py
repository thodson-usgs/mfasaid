from distutils.core import setup

setup(
    name='said',
    version='2.0.0dev1',
    packages=['said'],
    url='https://github.com/mdomanski-usgs/mfasaid',
    license='CC0 1.0',
    author='Marian Domanski',
    author_email='mdomanski@usgs.gov',
    description='Surrogate model creation',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python 3.6'
    ],
    python_requires='>=3', 
	install_requires=['matplotlib', 'numpy', 'pandas', 'statsmodels', 'scipy', 'linearmodel']
)
