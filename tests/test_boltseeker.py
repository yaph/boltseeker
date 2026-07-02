"""Tests for boltseeker._blob_qualifies."""

import cv2
import numpy as np

from boltseeker.boltseeker import _blob_qualifies

HEIGHT, WIDTH = 240, 320


def make_mask_buffer() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH), dtype=np.uint8)


def make_rect_contour(x: int, y: int, w: int, h: int) -> np.ndarray:
    """Return a closed rectangular contour as an OpenCV-compatible array."""
    return np.array(
        [
            [[x, y]],
            [[x + w, y]],
            [[x + w, y + h]],
            [[x, y + h]],
        ],
        dtype=np.int32,
    )


def make_gray(brightness: int = 10, rect: tuple | None = None, rect_brightness: int = 200) -> np.ndarray:
    """Return a grayscale frame, optionally with a bright rectangle."""
    frame = np.full((HEIGHT, WIDTH), brightness, dtype=np.uint8)
    if rect is not None:
        x, y, w, h = rect
        frame[y : y + h, x : x + w] = rect_brightness
    return frame


def contour_mean(contour: np.ndarray, gray: np.ndarray) -> float:
    """Return the mean luminance within a contour mask as OpenCV computes it."""
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    return cv2.mean(gray, mask=mask)[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBlobQualifies:
    def test_qualifies_when_large_and_bright(self):
        gray = make_gray(rect=(50, 50, 100, 100))
        contour = make_rect_contour(50, 50, 100, 100)
        assert _blob_qualifies(contour, gray, min_blob_area=100, luminance_threshold=50, mask_buffer=make_mask_buffer())

    def test_rejects_below_min_blob_area(self):
        gray = make_gray(rect=(50, 50, 100, 100))
        contour = make_rect_contour(50, 50, 100, 100)
        assert not _blob_qualifies(
            contour, gray, min_blob_area=999_999, luminance_threshold=50, mask_buffer=make_mask_buffer()
        )

    def test_rejects_below_luminance_threshold(self):
        gray = make_gray(rect=(50, 50, 100, 100), rect_brightness=120)
        contour = make_rect_contour(50, 50, 100, 100)
        assert not _blob_qualifies(
            contour, gray, min_blob_area=100, luminance_threshold=180, mask_buffer=make_mask_buffer()
        )

    def test_exact_min_blob_area_boundary(self):
        """A contour whose area equals min_blob_area should qualify (strict < comparison)."""
        contour = make_rect_contour(50, 50, 10, 10)
        area = int(cv2.contourArea(contour))  # 100 for a 10x10 rect
        gray = make_gray(rect=(50, 50, 10, 10))
        assert _blob_qualifies(
            contour, gray, min_blob_area=area, luminance_threshold=50, mask_buffer=make_mask_buffer()
        )

    def test_just_below_min_blob_area_boundary(self):
        """A contour whose area is one below min_blob_area should be rejected."""
        contour = make_rect_contour(50, 50, 10, 10)
        area = int(cv2.contourArea(contour))  # 100 for a 10x10 rect
        gray = make_gray(rect=(50, 50, 10, 10))
        assert not _blob_qualifies(
            contour, gray, min_blob_area=area + 1, luminance_threshold=50, mask_buffer=make_mask_buffer()
        )

    def test_exact_luminance_threshold_boundary(self):
        """Mean luminance equal to the threshold should pass (>= comparison).

        The threshold is derived from cv2.mean with the filled contour mask to
        match the value the function computes internally.
        """
        gray = make_gray(brightness=0, rect=(50, 50, 100, 100), rect_brightness=100)
        contour = make_rect_contour(50, 50, 100, 100)
        threshold = int(contour_mean(contour, gray))
        assert _blob_qualifies(
            contour, gray, min_blob_area=100, luminance_threshold=threshold, mask_buffer=make_mask_buffer()
        )

    def test_just_above_luminance_threshold_rejects(self):
        """A threshold one above the actual mean should be rejected."""
        gray = make_gray(brightness=0, rect=(50, 50, 100, 100), rect_brightness=100)
        contour = make_rect_contour(50, 50, 100, 100)
        threshold = int(contour_mean(contour, gray)) + 1
        assert not _blob_qualifies(
            contour, gray, min_blob_area=100, luminance_threshold=threshold, mask_buffer=make_mask_buffer()
        )

    def test_mask_buffer_cleared_between_calls(self):
        """A stale mask buffer from a previous call must not affect the result."""
        gray_bright = make_gray(rect=(50, 50, 100, 100))
        contour = make_rect_contour(50, 50, 100, 100)
        mask_buffer = make_mask_buffer()
        # First call leaves mask_buffer dirty
        _blob_qualifies(contour, gray_bright, min_blob_area=100, luminance_threshold=50, mask_buffer=mask_buffer)
        # Second call: small contour on a dark frame must not inherit the previous mask
        small_contour = make_rect_contour(50, 50, 2, 2)
        gray_dark = make_gray()
        assert not _blob_qualifies(
            small_contour, gray_dark, min_blob_area=100, luminance_threshold=50, mask_buffer=mask_buffer
        )
