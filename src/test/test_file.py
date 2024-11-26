"""
Author: Łukasz Kryczka

Module to store information about a test file.
Used to automate the testing process.
"""


class TestFile:
    """
    Class to store information about a test file.
    """

    def __init__(self, file_path: str, file_name: str, file_path_out: str) -> None:
        self.file_path = file_path
        self.file_name = file_name
        self.file_path_out = file_path_out

    def get_transformed_file_name(self, transformation_name: str) -> str:
        """
        Get the name of the transformed file.
        Obtained by appending the transformation name to the original file name.
        """
        extension = self.file_name.split('.')[-1]
        tmp_file_name = self.file_name.replace(f".{extension}", '')
        return f"{tmp_file_name}_{transformation_name}.{extension}"

    def get_transformed_file_path_out(self, transformation_name: str) -> str:
        """
        Get the path to the transformed file.
        Obtained by removing the name from the original file path and
        appending the transformed file name.

        :param transformation_name: Name of the transformation applied to the file
        """
        new_path = self.file_path_out.replace(self.file_name, '')
        return new_path + self.get_transformed_file_name(transformation_name)
