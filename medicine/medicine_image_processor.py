from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ============================================================
# PROCESSED MEDICINE IMAGES
# ============================================================

@dataclass
class ProcessedMedicineImages:
    """
    Different image representations of the same medicine crop.

    Each representation is intended for a different task.

    original:
        Original BGR medicine crop.

    enlarged:
        Upscaled BGR version.

    gray:
        Grayscale version.

    clahe:
        CLAHE-enhanced grayscale image.

    denoised:
        Mildly denoised grayscale image.

    sharpened:
        Contrast-enhanced + sharpened image.

    adaptive_threshold:
        OCR-friendly binary image.

    otsu_threshold:
        Alternative OCR-friendly binary image.

    inverted_threshold:
        Useful when medicine text is light-on-dark or
        reflective packaging reverses contrast.
    """

    original: np.ndarray

    enlarged: np.ndarray

    gray: np.ndarray

    clahe: np.ndarray

    denoised: np.ndarray

    sharpened: np.ndarray

    adaptive_threshold: np.ndarray

    otsu_threshold: np.ndarray

    inverted_threshold: np.ndarray


# ============================================================
# MEDICINE IMAGE PROCESSOR
# ============================================================

class MedicineImageProcessor:
    """
    Lightweight preprocessing for medicine packaging.

    Designed for:

        - reflective blister foil
        - medicine boxes
        - small printed medicine names
        - dosage text
        - uneven room lighting
        - moderate camera blur

    IMPORTANT:

    DINOv2 should continue using the ORIGINAL image.

    OCR can use:
        enlarged
        clahe
        sharpened
        adaptive_threshold
        otsu_threshold

    ORB can use:
        clahe
        sharpened

    The processor intentionally avoids expensive neural
    operations so it can run quickly during live recognition.
    """

    def __init__(
        self,
        upscale_factor: float = 2.0,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
    ) -> None:

        self.upscale_factor = max(
            1.0,
            float(
                upscale_factor
            ),
        )

        self.clahe_clip_limit = max(
            0.1,
            float(
                clahe_clip_limit
            ),
        )

        self.clahe_grid_size = max(
            2,
            int(
                clahe_grid_size
            ),
        )

        # ----------------------------------------------------
        # CREATE CLAHE ONCE
        #
        # Do not recreate this for every live frame.
        # ----------------------------------------------------

        self._clahe = cv2.createCLAHE(
            clipLimit=(
                self.clahe_clip_limit
            ),

            tileGridSize=(
                self.clahe_grid_size,
                self.clahe_grid_size,
            ),
        )


    # ========================================================
    # VALIDATE IMAGE
    # ========================================================

    @staticmethod
    def _validate_image(
        image: np.ndarray,
    ) -> None:

        if image is None:

            raise ValueError(
                "Medicine image cannot be None."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):

            raise TypeError(
                "Medicine image must be a NumPy array."
            )

        if image.size == 0:

            raise ValueError(
                "Medicine image cannot be empty."
            )

        if (
            image.ndim != 3
            or image.shape[2] != 3
        ):

            raise ValueError(
                (
                    "Medicine image must be a "
                    "BGR color image."
                )
            )


    # ========================================================
    # UPSCALE
    # ========================================================

    def upscale(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        self._validate_image(
            image
        )

        if (
            self.upscale_factor
            <= 1.0
        ):

            return image.copy()

        height, width = (
            image.shape[:2]
        )

        new_width = max(
            1,
            int(
                width
                * self.upscale_factor
            ),
        )

        new_height = max(
            1,
            int(
                height
                * self.upscale_factor
            ),
        )

        return cv2.resize(
            image,
            (
                new_width,
                new_height,
            ),
            interpolation=(
                cv2.INTER_CUBIC
            ),
        )


    # ========================================================
    # GRAYSCALE
    # ========================================================

    @staticmethod
    def to_gray(
        image: np.ndarray,
    ) -> np.ndarray:

        MedicineImageProcessor._validate_image(
            image
        )

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )


    # ========================================================
    # CLAHE
    # ========================================================

    def apply_clahe(
        self,
        gray: np.ndarray,
    ) -> np.ndarray:

        if gray is None:

            raise ValueError(
                "Grayscale image cannot be None."
            )

        if not isinstance(
            gray,
            np.ndarray,
        ):

            raise TypeError(
                "Grayscale image must be a NumPy array."
            )

        if gray.size == 0:

            raise ValueError(
                "Grayscale image cannot be empty."
            )

        if gray.ndim != 2:

            raise ValueError(
                "CLAHE requires a grayscale image."
            )

        return self._clahe.apply(
            gray
        )


    # ========================================================
    # DENOISE
    # ========================================================

    @staticmethod
    def denoise(
        gray: np.ndarray,
    ) -> np.ndarray:
        """
        Mild bilateral filtering.

        It reduces small noise while attempting to preserve
        text edges better than heavy Gaussian smoothing.
        """

        if (
            gray is None
            or gray.size == 0
        ):

            raise ValueError(
                "Image cannot be empty."
            )

        return cv2.bilateralFilter(
            gray,
            d=5,
            sigmaColor=35,
            sigmaSpace=35,
        )


    # ========================================================
    # SHARPEN
    # ========================================================

    @staticmethod
    def sharpen(
        gray: np.ndarray,
    ) -> np.ndarray:
        """
        Mild unsharp masking.

        We deliberately avoid aggressive sharpening because
        blister foil already contains many strong reflections.
        """

        if (
            gray is None
            or gray.size == 0
        ):

            raise ValueError(
                "Image cannot be empty."
            )

        blurred = cv2.GaussianBlur(
            gray,
            (
                0,
                0,
            ),
            sigmaX=1.0,
        )

        sharpened = cv2.addWeighted(
            gray,
            1.6,
            blurred,
            -0.6,
            0,
        )

        return sharpened


    # ========================================================
    # ADAPTIVE THRESHOLD
    # ========================================================

    @staticmethod
    def adaptive_threshold(
        gray: np.ndarray,
    ) -> np.ndarray:

        if (
            gray is None
            or gray.size == 0
        ):

            raise ValueError(
                "Image cannot be empty."
            )

        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )


    # ========================================================
    # OTSU THRESHOLD
    # ========================================================

    @staticmethod
    def otsu_threshold(
        gray: np.ndarray,
    ) -> np.ndarray:

        if (
            gray is None
            or gray.size == 0
        ):

            raise ValueError(
                "Image cannot be empty."
            )

        (
            _threshold,
            binary,
        ) = cv2.threshold(
            gray,
            0,
            255,
            (
                cv2.THRESH_BINARY
                + cv2.THRESH_OTSU
            ),
        )

        return binary


    # ========================================================
    # INVERT
    # ========================================================

    @staticmethod
    def invert(
        image: np.ndarray,
    ) -> np.ndarray:

        if (
            image is None
            or image.size == 0
        ):

            raise ValueError(
                "Image cannot be empty."
            )

        return cv2.bitwise_not(
            image
        )


    # ========================================================
    # PROCESS
    # ========================================================

    def process(
        self,
        image: np.ndarray,
    ) -> ProcessedMedicineImages:
        """
        Generate all useful lightweight representations.

        Processing sequence:

            original
                ↓
            upscale
                ↓
            grayscale
                ↓
            CLAHE
                ↓
            mild denoise
                ↓
            sharpen
                ↓
            threshold alternatives

        This function performs no OCR and no neural inference.
        """

        self._validate_image(
            image
        )

        original = (
            image.copy()
        )

        enlarged = (
            self.upscale(
                image
            )
        )

        gray = (
            self.to_gray(
                enlarged
            )
        )

        clahe = (
            self.apply_clahe(
                gray
            )
        )

        denoised = (
            self.denoise(
                clahe
            )
        )

        sharpened = (
            self.sharpen(
                denoised
            )
        )

        adaptive = (
            self.adaptive_threshold(
                sharpened
            )
        )

        otsu = (
            self.otsu_threshold(
                sharpened
            )
        )

        inverted = (
            self.invert(
                adaptive
            )
        )

        return ProcessedMedicineImages(
            original=(
                original
            ),

            enlarged=(
                enlarged
            ),

            gray=(
                gray
            ),

            clahe=(
                clahe
            ),

            denoised=(
                denoised
            ),

            sharpened=(
                sharpened
            ),

            adaptive_threshold=(
                adaptive
            ),

            otsu_threshold=(
                otsu
            ),

            inverted_threshold=(
                inverted
            ),
        )


    # ========================================================
    # FAST OCR VARIANTS
    # ========================================================

    def get_ocr_variants(
        self,
        image: np.ndarray,
    ) -> list[
        tuple[
            str,
            np.ndarray,
        ]
    ]:
        """
        Return only OCR-useful images.

        IMPORTANT:

        We are NOT going to OCR all of these continuously.

        The OCR module will stop as soon as sufficiently
        strong medicine text is found.
        """

        processed = (
            self.process(
                image
            )
        )

        return [
            (
                "enlarged",
                processed.enlarged,
            ),
            (
                "clahe",
                processed.clahe,
            ),
            (
                "sharpened",
                processed.sharpened,
            ),
            (
                "adaptive",
                processed.adaptive_threshold,
            ),
            (
                "otsu",
                processed.otsu_threshold,
            ),
        ]


    # ========================================================
    # FAST ORB IMAGE
    # ========================================================

    def get_orb_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Return one consistent enhanced grayscale image for ORB.

        We intentionally do not return many variants because
        ORB must remain fast in the live pipeline.
        """

        processed = (
            self.process(
                image
            )
        )

        return (
            processed.sharpened
        )