import zipfile
from pathlib import Path

def load_image_data(source_zip_file, destination_dir, new_dir = False):
    '''
    Extract zip image data into destination directory without changing the original
    folder structure.

    Args:
        source_zip_file (str): path to zip file
        destination_dir(str): target directory to send data
        new dir (bool, optional): Set to True to create a new directory to import image data. False by default.

        Returns:
            None
        Raises: 
            zipfile.BadZipFile: If zip file DNE or is corrupted
            Exception: Other unhandled exceptions
    '''


    zip_source = Path(source_zip_file)
    target_directory = Path(destination_dir)


    try:
        if new_dir:
            #Create new directory for data if specified
            target_directory.mkdir(parents=True, exist_ok=True)
            print(f"Data imported to new directory: {target_directory}")
        
        else:
            print(f"Importing data to existing directory: {target_directory}")

        #import data to target directory
        with zipfile.ZipFile(source_zip_file, 'r') as zip_ref:
            zip_ref.extractall(target_directory)
        print(f"Data successfully imported to: {target_directory}")

    except zipfile.BadZipFile:
        print("Error. File is not a Zipfile or may be corrupted.")
    except Exception as e:
        print(f"Error: {e}")
