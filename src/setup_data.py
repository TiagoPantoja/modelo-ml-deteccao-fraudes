import kagglehub
import shutil
import os


def setup_dataset():
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    source_file = f"{path}/creditcard.csv"

    destination_dir = "data"
    destination_file = os.path.join(destination_dir, "creditcard.csv")

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    shutil.copy(source_file, destination_file)


if __name__ == "__main__":
    setup_dataset()