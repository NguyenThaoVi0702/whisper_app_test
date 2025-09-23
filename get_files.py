import os
import shutil

def move_wav_files_with_phrase(source_folder, destination_folder, phrase):
    """
    Moves all .wav files containing a specific phrase in their name
    from a source folder to a destination folder.

    Args:
        source_folder (str): The path to the folder to search for .wav files.
        destination_folder (str): The path to the folder where matching files will be moved.
        phrase (str): The phrase to search for in the filenames.
    """
    # Create the destination folder if it doesn't already exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder: {destination_folder}")

    # Get a list of all files in the source folder
    try:
        files = os.listdir(source_folder)
    except FileNotFoundError:
        print(f"Error: The source folder '{source_folder}' was not found.")
        return

    # Loop through each file in the source folder
    for filename in files:
        # Check if the file is a .wav file and contains the specified phrase
        if filename.lower().endswith(".wav") and phrase.lower() in filename.lower():
            # Construct the full paths for the source and destination files
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)

            # Copy the file to the destination folder
            try:
                shutil.copy2(source_path, destination_path)
                print(f"Copied: {filename}")
            except Exception as e:
                print(f"Error copying {filename}: {e}")

if __name__ == "__main__":
    # Get user input for the folder paths and the search phrase
    source_directory = input("Enter the path to the source folder: ")
    destination_directory = input("Enter the path to the destination folder: ")
    search_phrase = input("Enter the phrase to search for in the .wav filenames: ")

    # Call the function to move the files
    move_wav_files_with_phrase(source_directory, destination_directory, search_phrase)

    print("\nProcess complete.")
