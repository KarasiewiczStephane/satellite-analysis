"""Automated EuroSAT dataset download and sample dataset creation."""

import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from src.utils.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DownloadProgressBar(tqdm):
    """Progress bar wrapper for ``urllib.request.urlretrieve``."""

    def update_to(
        self,
        b: int = 1,
        bsize: int = 1,
        tsize: int | None = None,
    ) -> None:
        """Update progress bar.

        Args:
            b: Number of blocks transferred so far.
            bsize: Size of each block in bytes.
            tsize: Total size in bytes (``None`` if unknown).
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_eurosat(output_dir: str | Path | None = None) -> Path:
    """Download and extract the EuroSAT dataset.

    Args:
        output_dir: Destination directory. Defaults to the configured ``data.raw_dir``.

    Returns:
        Path to the extracted dataset root.
    """
    output_dir = Path(output_dir or config.get("data.raw_dir", "data/raw"))
    output_dir.mkdir(parents=True, exist_ok=True)

    url = config.get("data.eurosat_url")
    zip_path = output_dir / "EuroSAT.zip"
    extract_path = output_dir / "EuroSAT"

    if extract_path.exists():
        logger.info("EuroSAT already exists at %s", extract_path)
        return extract_path

    logger.info("Downloading EuroSAT from %s", url)
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1) as progress:
        urllib.request.urlretrieve(url, zip_path, reporthook=progress.update_to)

    logger.info("Extracting to %s", extract_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    zip_path.unlink()
    return extract_path


def create_sample_dataset(
    source_dir: Path,
    sample_dir: Path,
    samples_per_class: int = 10,
) -> None:
    """Create a small sample dataset for testing and development.

    Args:
        source_dir: Root of the full EuroSAT dataset.
        sample_dir: Destination for the sampled subset.
        samples_per_class: Number of images to copy per class.
    """
    classes = config.get("data.classes")
    sample_dir.mkdir(parents=True, exist_ok=True)

    for class_name in classes:
        class_source = source_dir / class_name
        class_dest = sample_dir / class_name
        class_dest.mkdir(parents=True, exist_ok=True)

        if not class_source.exists():
            logger.warning("Source class directory not found: %s", class_source)
            continue

        images = list(class_source.glob("*.tif"))
        if not images:
            images = list(class_source.glob("*.jpg"))
        selected = random.sample(images, min(samples_per_class, len(images)))

        for img in selected:
            shutil.copy(img, class_dest / img.name)

    logger.info(
        "Created sample dataset with %d images per class at %s",
        samples_per_class,
        sample_dir,
    )
