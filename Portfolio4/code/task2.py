
import os
import random
from typing import List, Tuple

import cv2 

try:
    import detectron2
    from detectron2.engine import DefaultTrainer, DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.utils.visualizer import Visualizer
    
except ImportError as e:
    detectron2 = None

try:
    from labelme2coco import get_coco_json
except ImportError:
    # If labelme2coco is not available, conversion functions won't work.
    get_coco_json = None

# Config
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, os.pardir))
LABELME_DIR = os.path.join(DATA_DIR, 'extracted-log-labelled')
CONVERTED_DIR = os.path.join(DATA_DIR, 'extracted-converted-log-labelled')
COCO_TRAIN_JSON = os.path.join(CONVERTED_DIR, 'train_coco.json')
COCO_TEST_JSON = os.path.join(CONVERTED_DIR, 'test_coco.json')

# Where to save detection results
RCNN_OUTPUT_DIR = os.path.join(DATA_DIR, 'rcnn_test')


def convert_annotations(labelme_folder: str, train_json_path: str, test_json_path: str,
                        test_ratio: float = 0.2) -> None:

    if get_coco_json is None:
        raise RuntimeError(
            "labelme2coco is not installed. Install it with `pip install labelme2coco`."
        )
    # Collect all JSON files in the directory
    annotation_files = [os.path.join(labelme_folder, f) for f in os.listdir(labelme_folder)
                        if f.endswith('.json')]
    random.shuffle(annotation_files)
    split_index = int(len(annotation_files) * (1 - test_ratio))
    train_files = annotation_files[:split_index]
    test_files = annotation_files[split_index:]

    def write_coco(files: List[str], output_file: str) -> None:
        data = get_coco_json(files)
        with open(output_file, 'w') as f:
            f.write(data)

    write_coco(train_files, train_json_path)
    write_coco(test_files, test_json_path)


def register_coco_datasets(train_json: str, test_json: str, image_root: str) -> Tuple[str, str]:

    from detectron2.data.datasets import register_coco_instances
    train_name = 'log_train_dataset'
    test_name = 'log_test_dataset'
    register_coco_instances(train_name, {}, train_json, image_root)
    register_coco_instances(test_name, {}, test_json, image_root)
    return train_name, test_name


def build_and_train_model(dataset_name: str, num_epochs: int = 10) -> str:

    cfg = get_cfg()
    # Load a standard configuration and make minimal changes
    from detectron2.model_zoo import get_config_file
    from detectron2.model_zoo import get_checkpoint_url
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = num_epochs * 100  # approximate: 100 iterations per epoch
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only one class: log

    output_dir = os.path.join(PROJECT_ROOT, 'detectron2_output')
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    return cfg.OUTPUT_DIR


def perform_inference(cfg_dir: str, test_dataset_name: str, output_dir: str) -> None:

    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(cfg_dir, 'config.yaml'))
    cfg.MODEL.WEIGHTS = os.path.join(cfg_dir, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get(test_dataset_name)
    dataset_dicts = DatasetCatalog.get(test_dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for item in dataset_dicts:
        file_name = item["file_name"]
        image = cv2.imread(file_name)
        outputs = predictor(image)

        v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # Save the visualised image in BGR format (OpenCV uses BGR ordering)
        out_img = out.get_image()[:, :, ::-1]
        base_name = os.path.basename(file_name)
        cv2.imwrite(os.path.join(output_dir, base_name), out_img)

    print(f"Saved detection results to {output_dir}")


def count_detected_logs(prediction_folder: str) -> None:

    for fname in os.listdir(prediction_folder):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(prediction_folder, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        # Convert to grayscale and threshold to segment masks; this is a
        # simplistic way to count distinct coloured blobs.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_logs = len(contours)
        print(f"{fname}: {num_logs} detected logs")


def main() -> None:
    """Entry point for Mask R‑CNN training and inference.

    This function performs the full pipeline when executed.  It assumes that
    the LabelMe annotations have already been converted to COCO format and
    split into training and testing JSON files.  If this is not the case,
    uncomment the `convert_annotations` call and supply the correct
    directories.
    """
    if detectron2 is None:
        raise RuntimeError(
            "Detectron2 is not installed. Please install it before running this script."
        )

    # If JSON files do not yet exist, convert them from LabelMe below
    # convert_annotations(LABELME_DIR, COCO_TRAIN_JSON, COCO_TEST_JSON, test_ratio=0.2)

    # Register datasets
    train_name, test_name = register_coco_datasets(COCO_TRAIN_JSON, COCO_TEST_JSON, LABELME_DIR)

    # Train
    cfg_dir = build_and_train_model(train_name, num_epochs=10)

    # Inference
    perform_inference(cfg_dir, test_name, RCNN_OUTPUT_DIR)

    # Count the detected objects
    count_detected_logs(RCNN_OUTPUT_DIR)


if __name__ == '__main__':
    main()