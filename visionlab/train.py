"""Reproducible YOLO fine-tuning from the terminal, never from web uploads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile
from .config import ROOT

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
DEMO_URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip'


def download_demo(folder: Path) -> Path:
    import yaml
    import ultralytics
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / 'coco8.zip'
    with urllib.request.urlopen(DEMO_URL, timeout=30) as response:
        data = response.read(20 * 1024 * 1024 + 1)
    if len(data) > 20 * 1024 * 1024:
        raise ValueError('The demo download is unexpectedly large.')
    archive.write_bytes(data)
    with zipfile.ZipFile(archive) as bundle:
        if sum(item.file_size for item in bundle.infolist()) > 50 * 1024 * 1024:
            raise ValueError('The expanded demo is unexpectedly large.')
        for item in bundle.infolist():
            if not (folder / item.filename).resolve().is_relative_to(folder.resolve()):
                raise ValueError('Unsafe dataset archive path.')
        bundle.extractall(folder)
    source = Path(ultralytics.__file__).parent / 'cfg' / 'datasets' / 'coco8.yaml'
    names = yaml.safe_load(source.read_text(encoding='utf-8'))['names']
    config = {'path': str((folder / 'coco8').resolve()), 'train': 'images/train', 'val': 'images/val', 'names': names}
    destination = folder / 'demo.yaml'
    destination.write_text(yaml.safe_dump(config), encoding='utf-8')
    return destination


def validate_dataset(path: Path) -> tuple[dict, dict]:
    import yaml
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or not {'train', 'val', 'names'} <= config.keys():
        raise ValueError('Dataset YAML needs train, val and names.')
    if 'download' in config:
        raise ValueError('Download scripts are not accepted. Prepare the dataset directories first.')
    names = config['names']
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or set(names) != set(range(len(names))) or not names:
        raise ValueError('Class names must use consecutive IDs beginning at zero.')
    if not all(isinstance(name, str) and name.strip() for name in names.values()):
        raise ValueError('Every class needs a nonempty name.')
    root = Path(config.get('path', '.'))
    if not root.is_absolute():
        root = path.parent / root
    sanitized = {'path': str(root.resolve()), 'names': names}
    fingerprints, counts = {}, {}
    for split in ('train', 'val'):
        if not isinstance(config[split], str):
            raise ValueError('This trainer expects train and val image directories.')
        directory = Path(config[split])
        if not directory.is_absolute():
            directory = root / directory
        directory = directory.resolve()
        images = sorted(p for p in directory.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
        if not directory.is_dir() or not images:
            raise ValueError(f'No images found in the {split} directory.')
        fingerprints[split] = {hashlib.sha256(p.read_bytes()).hexdigest() for p in images}
        counts[split] = len(images)
        sanitized[split] = str(directory)
    if fingerprints['train'] & fingerprints['val']:
        raise ValueError('Training and validation must not contain duplicate images.')
    return sanitized, counts


def train_model(dataset: Path, output: Path, epochs=5, image_size=320) -> dict:
    import yaml
    import torch
    from ultralytics import YOLO
    from .detection import load_detector
    if not 1 <= epochs <= 100 or image_size not in {320, 640, 960}:
        raise ValueError('Use 1–100 epochs and image size 320, 640 or 960.')
    config, counts = validate_dataset(dataset)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'best.pt').exists():
        raise ValueError('This output already contains best.pt. Choose a new output directory.')
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='visionlab-train-') as temporary:
        clean_yaml = Path(temporary) / 'dataset.yaml'
        clean_yaml.write_text(yaml.safe_dump(config), encoding='utf-8')
        baseline = load_detector('yolo11n')
        validation = dict(data=str(clean_yaml), imgsz=image_size, batch=2, workers=0, device='cpu',
                          plots=False, verbose=False, project=str(output / 'runs'))
        before = baseline.val(**validation, name='baseline')
        before_metrics = {k: float(v) for k, v in before.results_dict.items()}
        model = YOLO(baseline.ckpt_path)
        model.train(data=str(clean_yaml), epochs=epochs, imgsz=image_size, batch=2, workers=0,
                    device='cpu', seed=42, deterministic=True, amp=False, optimizer='AdamW',
                    lr0=0.0001, freeze=10, plots=False, save=True, project=str(output / 'runs'),
                    name='finetune', exist_ok=False, patience=epochs + 1, verbose=False)
        shutil.copy2(Path(model.trainer.best), output / 'best.pt')
        trained = YOLO(str(output / 'best.pt'))
        after = trained.val(**validation, name='trained')
        report = {'base_model': 'YOLO11n', 'epochs': epochs, 'image_size': image_size, 'seed': 42,
                  'device': 'cpu', 'freeze_layers': 10, 'learning_rate': 0.0001, 'images': counts,
                  'baseline_validation': before_metrics,
                  'trained_validation': {k: float(v) for k, v in after.results_dict.items()},
                  'checkpoint_sha256': hashlib.sha256((output / 'best.pt').read_bytes()).hexdigest(),
                  'checkpoint_bytes': (output / 'best.pt').stat().st_size,
                  'note': 'Validation selects the checkpoint; this is not an independent test set. A small demo does not establish general accuracy.'}
        (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo', action='store_true', help='Download the eight-image COCO8 demonstration dataset.')
    source.add_argument('--data', type=Path, help='Local YOLO YAML with distinct train/val image directories.')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--image-size', type=int, default=320, choices=[320, 640, 960])
    parser.add_argument('--output', type=Path, default=ROOT / 'training' / 'my-run')
    args = parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(prefix='visionlab-demo-') as temporary:
            dataset = download_demo(Path(temporary)) if args.demo else args.data.resolve()
            report = train_model(dataset, args.output.resolve(), args.epochs, args.image_size)
        print(json.dumps(report, indent=2))
        print('Saved best.pt and metrics.json. Compare validation results before using these weights.')
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        parser.exit(1, f'Training failed: {error}\n')


if __name__ == '__main__':
    main()
