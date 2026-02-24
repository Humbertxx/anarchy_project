from pathlib import Path

############# UTILS CLASS FOR FILES CHANGES #############

DATA_DIR = Path.home() / "Python" / "file_processor" / "data"

## creates directory if not exist, rewrites if exists  
def directory_creation():
    for folder in [DATA_DIR]:
        folder.mkdir(parents=True, exist_ok=True)  