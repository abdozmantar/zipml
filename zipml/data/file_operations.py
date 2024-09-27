import os
import pandas as pd
import zipfile
import logging
from typing import Optional, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def walk_through_dir(dir_path: str) -> pd.DataFrame:
    """
    Walks through dir_path returning its contents as a Pandas DataFrame.

    Args:
        dir_path (str): Target directory.
  
    Returns:
        pd.DataFrame: A DataFrame containing dirpath, dirnames, and filenames.
    """
    # Initialize lists to store directory information
    dir_paths = []  # To store the path of each directory
    dir_names = []  # To store the names of subdirectories
    file_names = []  # To store the names of files (images)

    # Walk through the directory structure
    for dirpath, dirnames, filenames in os.walk(dir_path):
        # Append the collected information to the lists
        dir_paths.append(dirpath)
        dir_names.append(dirnames)
        file_names.append(filenames)

        # Log the number of subdirectories and images in the current directory
        logging.info(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

    # Create a DataFrame from the lists
    df = pd.DataFrame({
        'dir_path': dir_paths,
        'dir_names': dir_names,
        'file_names': file_names
    })

    return df  # Return the DataFrame

def unzip_data(filename: str, extract_path: Optional[str] = None) -> None:
    """
    Unzips the specified zip file into the current working directory or a specified path.

    Args:
        filename (str): The file path to the target zip file that needs to be unzipped.
        extract_path (Optional[str]): The directory path where the contents should be extracted.
                                       If None, the contents will be extracted to the current working directory.
    """
    # Open the specified zip file in read mode
    with zipfile.ZipFile(filename, "r") as zip_ref:
        # Extract all contents to the specified directory or current working directory
        zip_ref.extractall(extract_path if extract_path else ".")

def read_lines_from_file(filename: str) -> List[str]:
    """
    Reads the contents of a text file and returns the lines as a list of strings.

    Args:
        filename (str): A string containing the path to the target text file.

    Returns:
        List[str]: A list of strings, where each string represents a line from the file.
    """
    with open(filename, "r") as file:
        return file.readlines()  # Read all lines and return them as a list
    


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads a dataset from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        DataFrame: Loaded data as a pandas DataFrame.
    """
    logging.info(f"Loading dataset from {file_path}.")
    return pd.read_csv(file_path)



