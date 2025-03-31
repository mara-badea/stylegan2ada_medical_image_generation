import json

from custom_processors.dataset_processor import DatasetProcessor
import os


SCAN_TYPE_DICT = {"rmn": 0, "mri": 0, "xray": 1, "ct": 2}

BRAIN_DATASET_DISEASE_DICT = {
    "notumor": 0,
    "glioma": 1,
    "meningioma": 2,
    "pituitary": 3,
}

LUNGS_DATASET_DISEASE_DICT = {"normal": 0, "pneumonia": 4}


class DatasetProcessorFolderTag(DatasetProcessor):
    def __init__(
        self,
        dataset_path: str,
        scanned_organ: str,
        scan_type: str,
        process_type: str = "train",
    ):
        super().__init__(dataset_path, process_type)
        self.organ_label = 0 if scanned_organ.lower() == "brain" else 1
        self.scan_label = SCAN_TYPE_DICT[scan_type.lower()]
        self.class_dict = (
            BRAIN_DATASET_DISEASE_DICT
            if self.organ_label == 0
            else LUNGS_DATASET_DISEASE_DICT
        )
        self.class_labels = list(
            self._get_class_labels(self.set_path, self.class_dict)
        )
        self.all_labels = {
            "labels": self._process_labels(self.class_labels),
        }
        self.json_file_path = os.path.join(self.set_path, "dataset.json")
        self.save_labels_as_json(self.json_file_path, self.all_labels)

    def _process_labels(self, class_labels):
        labels_with_metadata = []
        for img_path, disease_label in class_labels:
            label = int(
                f"{str(disease_label)}{str(self.organ_label)}{str(self.scan_label)}"
            )
            labels_with_metadata.append([img_path, label])
        return labels_with_metadata

    def get_labels(self):
        return self.all_labels

    @staticmethod
    def _get_class_labels(set_path: str, class_dict: dict):
        data_folder = os.listdir(set_path)

        for tag in data_folder:
            image_folder_path = os.path.join(set_path, tag)
            images = os.listdir(image_folder_path)

            for image in images:
                image_absolute_path = os.path.join(image_folder_path, image)
                image_relative_path = os.path.relpath(
                    image_absolute_path, set_path
                ).replace("\\", "/")
                yield image_relative_path, class_dict[tag.lower()]

    @staticmethod
    def save_labels_as_json(path: str, labels: dict):
        json_file = json.dumps(labels, indent=4)
        with open(path, "w+") as f:
            f.write(json_file)
