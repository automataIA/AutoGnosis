import os
import re

def create_directories(index_lines):
    for line in index_lines:
        # Find the title by removing the leading whitespace
        title = re.sub(r"^\s+", "", line)
        # Replace ":" and spaces with "-", then create the directory
        dir_name = title.replace(":", "-").replace(" ", "-").replace("\n","")
        os.makedirs(dir_name, exist_ok=True)

def create_index_files(index_lines):
    for line in index_lines:
        # Find the title by removing the leading whitespace
        title = re.sub(r"^\s+", "", line)
        # Replace ":" and spaces with "-", then create the directory
        dir_name = title.replace(":", "-").replace(" ", "-").replace("\n","")
        # Create the file name and path
        file_name = "_index.it.md"
        file_path = os.path.join(dir_name, file_name)
        # Write the file contents
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f'title: "{title}"\n')
            f.write("date: 2022-12-28T18:27:41+01:00\n")
            f.write("draft: false\n")
            f.write("---\n")

# Read the input file
with open("index.txt", "r", encoding="utf-8") as f:
    index_lines = f.readlines()

# Create the directories and index files
create_directories(index_lines)
create_index_files(index_lines)
