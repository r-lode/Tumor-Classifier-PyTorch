import os
import zipfile

def load_data(source, destination)-> None: 
    '''
    Function to load zip data into working directory 

    Args:
        source: source directory containing zip files
        destination: working directory to load zip file data

    Returns:
        None 
    '''

    with zipfile.ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(destination)