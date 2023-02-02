import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="zeroxQA",
	version="0.6",
	author="Somnath Kumar",
	author_email="hexplex0xff@gmail.com",
	description="Code for Few Shot learning Open Domain Question answering for out domain themes",
	long_description=long_description,
	url="https://github.com/hex-plex/zeroxQA",
	packages=setuptools.find_packages(),
	install_requires=[""]
)
