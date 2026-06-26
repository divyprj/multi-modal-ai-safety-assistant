"""Utility functions for image processing and display formatting."""

import io
from PIL import Image


def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize an image while maintaining aspect ratio.
    
    Args:
        image: PIL Image to resize.
        max_size: Maximum dimension (width or height).
        
    Returns:
        Resized PIL Image.
    """
    width, height = image.size
    if max(width, height) <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def image_to_bytes(image: Image.Image, format: str = "JPEG", quality: int = 90) -> bytes:
    """Convert a PIL Image to bytes."""
    buffer = io.BytesIO()
    image_rgb = image.convert("RGB")
    image_rgb.save(buffer, format=format, quality=quality)
    return buffer.getvalue()


def bytes_to_image(data: bytes) -> Image.Image:
    """Convert bytes to a PIL Image."""
    return Image.open(io.BytesIO(data))


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining = seconds % 60
        return f"{minutes}m {remaining:.1f}s"


def get_severity_color(severity: str) -> str:
    """Get the display color for a severity level."""
    colors = {
        "CRITICAL": "#FF0040",
        "HIGH": "#FF6B00",
        "MEDIUM": "#FFD600",
        "LOW": "#00E676",
    }
    return colors.get(severity, "#FFFFFF")
