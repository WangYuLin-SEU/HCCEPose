"""Image sizes and padding shared by RGB inference and inverse mask mapping."""


def inference_image_geometry(height, width):
    """Return the resize divisor, scaled and padded H/W, and top/left padding."""
    if height <= 0 or width <= 0:
        raise ValueError('Image height and width must be positive')
    ratio = max(height, width) / 640
    if ratio > 1:
        scaled_h = max(1, int(height / ratio))
        scaled_w = max(1, int(width / ratio))
    else:
        scaled_h, scaled_w = height, width
    padded_h = ((scaled_h + 31) // 32) * 32
    padded_w = ((scaled_w + 31) // 32) * 32
    top = (padded_h - scaled_h) // 2
    left = (padded_w - scaled_w) // 2
    return ratio, scaled_h, scaled_w, padded_h, padded_w, top, left
