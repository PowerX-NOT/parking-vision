def classify_occupancy(frame, slots):
    """
    Each YOLO-detected box is 'occupied'.
    If you have a total slot count, you can infer vacant = total - occupied.
    """
    return ['occupied'] * len(slots)
