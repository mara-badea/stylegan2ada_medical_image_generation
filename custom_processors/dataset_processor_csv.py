import os
import json
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
from custom_processors.dataset_processor import DatasetProcessor

NIH_CLAHE_LUNG_DATASET_DISEASE_DICT = {
    "atelectasis": 0,
    "cardiomegaly": 1,
    "consolidation": 2,
    "edema": 3,
    "effusion": 4,
    "emphysema": 5,
    "fibrosis": 6,
    "hernia": 7,
    "infiltration": 8,
    "mass": 9,
    "no finding": 10,
    "nodule": 11,
    "pleural thickening": 12,
    "pneumonia": 13,
    "pneumothorax": 14
}

class DatasetProcessorCSV(DatasetProcessor):
    def __init__(self, dataset_path, label_csv_path, split_ratio=0.8, process_type="train"):
        self.label_csv_path = label_csv_path
        self.split_ratio = split_ratio
        self.full_labels = self._load_labels()
        self.labels = self.get_labels()
        self.train_files, self.test_files = self._split_files()
        self.file_list = self.train_files if process_type.lower() == "train" else self.test_files
        self.save_split(dataset_path, process_type)
        super().__init__(dataset_path, process_type)




    def _load_labels(self):
        df = pd.read_csv(self.label_csv_path)
        df.columns = [col.lower().replace("_", " ") if col != "Path" else col for col in df.columns]

        full_labels = {}
        for _, row in df.iterrows():
            for col in df.columns[1:]:
                if row[col] == 1 and col in NIH_CLAHE_LUNG_DATASET_DISEASE_DICT:
                    full_labels[row["Path"]] = [NIH_CLAHE_LUNG_DATASET_DISEASE_DICT[col]]
                    break
        return full_labels

    def _split_files(self):
        filenames = list(self.full_labels.keys())
        train_files, test_files = train_test_split(filenames, test_size=1 - self.split_ratio, random_state=42)
        return train_files, test_files

    def get_labels(self):
        return {"labels": [[fname, self.full_labels[fname][0]] for fname in self.full_labels.keys()]}

    def save_split(self, dataset_path, process_type, overwrite=False):
        split_name = "train" if process_type.lower() == "train" else "test"
        split_folder = os.path.join(dataset_path, split_name)
        image_dest = os.path.join(split_folder, "images")
        os.makedirs(image_dest, exist_ok=True)

        split_labels = []
        skipped_files = []

        for fname in self.file_list:
            src = os.path.join(dataset_path, fname)
            dst = os.path.join(image_dest, fname)

            dst_dir = os.path.dirname(dst)
            os.makedirs(dst_dir, exist_ok=True)

            if not os.path.exists(src):
                print(f"Missing source file: {fname}")
                continue

            if os.path.exists(dst) and not overwrite:
                skipped_files.append(fname)
            else:
                shutil.copy2(src, dst)

            split_labels.append([fname, self.full_labels[fname][0]])

        with open(os.path.join(split_folder, "dataset.json"), "w") as f:
            json.dump({"labels": split_labels}, f, indent=4)

        print(f"{split_name} split saved: {len(split_labels)} images")

        if skipped_files:
            print(f"Skipped {len(skipped_files)} existing files in '{split_name}' split:")
            for fname in skipped_files:
                print(f"   - {fname}")

