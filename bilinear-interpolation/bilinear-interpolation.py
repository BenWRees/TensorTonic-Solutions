def bilinear_resize(image: list, new_h: int, new_w: int) -> list:
    """
    Resize a 2D image using corner-aligned bilinear interpolation.
    """
    h, w = len(image), len(image[0])

    def interpolation_weights(source_size, target_size):
        for i in range(target_size):
            pos = i * (source_size - 1) / (target_size - 1) if target_size > 1 else 0
            low = int(pos)
            high = min(low + 1, source_size - 1)
            weight = pos - low
            yield low, high, weight

    rows = list(interpolation_weights(h, new_h))
    cols = list(interpolation_weights(w, new_w))

    result = []

    for y0, y1, dy in rows:
        row = []

        for x0, x1, dx in cols:
            top = image[y0][x0] * (1 - dx) + image[y0][x1] * dx
            bottom = image[y1][x0] * (1 - dx) + image[y1][x1] * dx

            row.append(top * (1 - dy) + bottom * dy)

        result.append(row)

    return result