import os
import shutil
from PIL import Image


# class label_with_autodistill():



def label_with_autodistill(images_path):
    def __init__(self):
        self.home_dir = os.getcwd()

    def x(self):
        images_dir = os.path.join(HOME, "images")

        try:
            os.mkdir(images_dir)
            print(f"Directory {images_dir} created successfully.")
        except FileExistsError:
            print(f"Directory {images_dir} already exists.")
        except Exception as e:
            print(f"Error creating directory: {e}")
            




    def clear_folders(self):
        # Specify the file or directory path to remove
        file_or_directory_path = os.path.join(self.home_dir, "runs")
        try:
            if os.path.isfile(file_or_directory_path):
                os.remove(file_or_directory_path)
                print(f"File {file_or_directory_path} removed successfully.")
            elif os.path.isdir(file_or_directory_path):
                shutil.rmtree(file_or_directory_path)
                print(f"Directory {file_or_directory_path} and its contents removed successfully.")
            else:
                print(f"Path {file_or_directory_path} does not exist.")
        except Exception as e:
            print(f"Error removing file or directory: {e}")

        file_or_directory_path = os.path.join(self.home_dir, "images")
        try:
            if os.path.isfile(file_or_directory_path):
                os.remove(file_or_directory_path)
                print(f"File {file_or_directory_path} removed successfully.")
            elif os.path.isdir(file_or_directory_path):
                shutil.rmtree(file_or_directory_path)
                print(f"Directory {file_or_directory_path} and its contents removed successfully.")
            else:
                print(f"Path {file_or_directory_path} does not exist.")
        except Exception as e:
            print(f"Error removing file or directory: {e}")

        file_or_directory_path = os.path.join(self.home_dir, "dataset")
        try:
            if os.path.isfile(file_or_directory_path):
                os.remove(file_or_directory_path)
                print(f"File {file_or_directory_path} removed successfully.")
            elif os.path.isdir(file_or_directory_path):
                shutil.rmtree(file_or_directory_path)
                print(f"Directory {file_or_directory_path} and its contents removed successfully.")
            else:
                print(f"Path {file_or_directory_path} does not exist.")
        except Exception as e:
            print(f"Error removing file or directory: {e}")

        file_or_directory_path = os.path.join(self.home_dir, "annotated_images")
        try:
            if os.path.isfile(file_or_directory_path):
                os.remove(file_or_directory_path)
                print(f"File {file_or_directory_path} removed successfully.")
            elif os.path.isdir(file_or_directory_path):
                shutil.rmtree(file_or_directory_path)
                print(f"Directory {file_or_directory_path} and its contents removed successfully.")
            else:
                print(f"Path {file_or_directory_path} does not exist.")
        except Exception as e:
            print(f"Error removing file or directory: {e}")
    
    




label_with_autodistill("")