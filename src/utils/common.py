import os
import pickle
import sys

import numpy as np
import yaml

from ..utils.exception import PhishingDetectionException


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Content of the YAML file.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise PhishingDetectionException(f"Error reading YAML file: {e}", sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes content to a YAML file.

    Args:
        file_path (str): Path to the YAML file.
        content (object): Content to write to the file.
        replace (bool): Whether to replace the file if it already exists.
    """
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise PhishingDetectionException(f"Error writing YAML file: {e}", sys)


def save_numpy_array_data(file_path: str, array: np.array) -> None:
    """
    Saves a NumPy array to a file.

    Args:
        file_path (str): Path to the file.
        array (np.array): NumPy array to save.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise PhishingDetectionException(f"Error saving NumPy array: {e}", sys)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Loads a NumPy array from a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        np.array: Loaded NumPy array.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise PhishingDetectionException(f"Error loading NumPy array: {e}", sys)


def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object to a file using pickle.

    Args:
        file_path (str): Path to the file.
        obj (object): Python object to save.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise PhishingDetectionException(f"Error saving object: {e}", sys)


def load_object(file_path: str) -> object:
    """
    Loads a Python object from a file using pickle.

    Args:
        file_path (str): Path to the file.

    Returns:
        object: Loaded Python object.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist.")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise PhishingDetectionException(f"Error loading object: {e}", sys)
