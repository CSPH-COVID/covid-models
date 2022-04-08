### Python Standard Library ###
import setuptools
### Third Party Imports ###
### Local Imports ###


# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(name='db-utils',
                 version='0.0.1',
                 description='Tools for interacting with the database.',
                 url='https://github.com/vanadata/covid-analysis/tree/main/db',
                 author='David Jacobson',
                 author_email='david@vanadata.io',
                 packages=setuptools.find_packages(),
                 zip_safe=False)
