import re
import shutil
import zipfile

from os import listdir
from os.path import isfile, join
from pathlib import Path
from importlib import resources

import logging

def save_zip(dir: Path, response):
    """save a zip file from a Flask Response into a local folder.
    
    Parameters
    ----------
    start: Path
        Absolute path of directory for extracting a Flask response's zipped contents into.

    response: Flask Response that contains a zipped file.

    Returns
    -------
    output_path
        Path to folder of unzipped contents.
    """

    # Saving zip file locally
    name = re.findall('filename=(.+)', response.headers['Content-Disposition'])[0]
    print(f"Saving response zip file {name} locally")
    with response as r, open(name,'wb') as out_file:
        shutil.copyfileobj(r, out_file)
    # Unzipping file 
    output_path = dir / name.split('.')[0]
    
    if output_path.exists() and output_path.is_dir():
        logging.getLogger('response_utils').warning(f"Folder {output_path} already exists. REPLACING existing folder.")
        shutil.rmtree(output_path)
        
    logging.getLogger('response_utils').info('Saving file to: {}'.format(output_path))
    with zipfile.ZipFile(name, 'r') as zip_ref:
        zip_ref.extractall(output_path)
    return output_path