from LabelsPreparator import save_labels_in_json
from ConstValues import (
    PROJECT_DIR,
    DATASET_DIR_NAME,
    LABELS_FILE_NAME,
    TRAINING_LABELS_DIR_NAME,
    VALIDATION_LABELS_DIR_NAME,
    TRAINING_DIRS,
    VALIDATION_DIRS,
)

save_labels_in_json(
    PROJECT_DIR,
    DATASET_DIR_NAME,
    TRAINING_LABELS_DIR_NAME,
    LABELS_FILE_NAME,
    classes_to_consider=TRAINING_DIRS,
)
save_labels_in_json(
    PROJECT_DIR,
    DATASET_DIR_NAME,
    VALIDATION_LABELS_DIR_NAME,
    LABELS_FILE_NAME,
    classes_to_consider=VALIDATION_DIRS,
)
