"""
Input and output validation helpers used by the Flask routes and services.
Keeping validation logic here (rather than scattered in view functions)
makes it independently unit-testable and reusable by the CLI scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from utils.image_utils import is_corrupted, read_image_bgr, compute_blur_score
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]

    def add_error(self, message: str) -> None:
        self.is_valid = False
        self.errors.append(message)


def validate_upload_request(
    has_file: bool,
    filename: str,
    allowed_extensions: frozenset,
    content_length: Optional[int],
    max_content_length: int,
) -> ValidationResult:
    """Validate the raw HTTP upload before it ever touches disk."""
    result = ValidationResult(is_valid=True, errors=[])

    if not has_file or not filename:
        result.add_error("No file was provided in the request.")
        return result

    if "." not in filename:
        result.add_error("File has no extension.")
    else:
        ext = filename.rsplit(".", 1)[1].lower()
        if ext not in allowed_extensions:
            result.add_error(
                f"Unsupported file type '.{ext}'. Allowed: "
                f"{', '.join(sorted(allowed_extensions))}"
            )

    if content_length is not None and content_length > max_content_length:
        max_mb = max_content_length / (1024 * 1024)
        result.add_error(f"File exceeds the {max_mb:.0f} MB upload limit.")

    return result


def validate_image_file(
    path: str, min_dimension: int = 64, blur_threshold: float = 30.0
) -> ValidationResult:
    """Deeper, content-level validation once the file is on disk: decode
    integrity, minimum resolution, and blur detection."""
    result = ValidationResult(is_valid=True, errors=[])

    if is_corrupted(path):
        result.add_error("The uploaded file is not a valid or readable image.")
        return result

    image = read_image_bgr(path)
    height, width = image.shape[:2]
    if height < min_dimension or width < min_dimension:
        result.add_error(
            f"Image resolution too low ({width}x{height}). "
            f"Minimum is {min_dimension}x{min_dimension}."
        )

    blur_score = compute_blur_score(image)
    if blur_score < blur_threshold:
        result.add_error(
            "Image appears too blurry for a reliable prediction "
            f"(sharpness score {blur_score:.1f} < {blur_threshold})."
        )

    return result


def validate_class_names(class_names: List[str]) -> ValidationResult:
    """Ensure the dataset/model has at least two discoverable classes
    before allowing training or prediction to proceed."""
    result = ValidationResult(is_valid=True, errors=[])
    if not class_names:
        result.add_error(
            "No dataset classes were discovered. Add class sub-folders "
            "under dataset/ (see dataset/README.md)."
        )
    elif len(class_names) < 2:
        result.add_error(
            "At least two classes are required for classification; "
            f"found only: {class_names}"
        )
    return result


def validate_prediction_output(
    probabilities: List[float], class_names: List[str]
) -> ValidationResult:
    """Sanity-check model output before it is returned to the client."""
    result = ValidationResult(is_valid=True, errors=[])
    if len(probabilities) != len(class_names):
        result.add_error(
            "Model output size does not match the number of known classes."
        )
        return result
    total = sum(probabilities)
    if not (0.98 <= total <= 1.02):
        result.add_error(
            f"Prediction probabilities do not sum to ~1.0 (got {total:.4f})."
        )
    return result
