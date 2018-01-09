#pylint: disable=C
from setuptools import setup, find_packages
import setuptools.command.install
from Cython.Build import cythonize
import requests
import os
import numpy as np


# code to download from google Drive
# copy from https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : file_id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : file_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
# end of the copied code


class PostInstallCommand(setuptools.command.install.install):
    """Post-installation for installation mode."""

    def run(self):
        setuptools.command.install.install.run(self)

        google_drive_file_id = '0B5e7DAOiLEZwSkdfXzBYT29Nc3c'
        destination = os.path.join(self.install_lib, 'lie_learn/representations/SO3/pinchon_hoggan/J_dense_0-278.npy')

        print("Start to download file ID {} from google drive into {}".format(google_drive_file_id, destination))

        try:
            download_file_from_google_drive(google_drive_file_id, destination)
        except:
            print("Error during the download")
            raise


setup(
    name='lie_learn',
    packages=find_packages(),
    ext_modules=cythonize('lie_learn/**/*.pyx'),
    cmdclass={ 'install': PostInstallCommand },
    include_dirs=[np.get_include()],
)
