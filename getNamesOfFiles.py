import os
def get_file_list(folder_path):
    file_list = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_list.append(str(file_name).split("_")[0])
            # file_list.append(str(file_name)[:-9]) # skeleton
    return file_list

# folder_path = "./data/raw/thermal"
folder_path = "/home/eslab/바탕화면/jangjange/data/raw/rgb"

files = get_file_list(folder_path)

with open("/home/eslab/바탕화면/jangjange/data/processed/samples_names.txt", 'w') as file:
    for name in files:
        file.write(str(name) + "\n")
