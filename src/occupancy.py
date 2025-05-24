def classify_occupancy(predictions):
    """
    Classify each slot as occupied or vacant based on predictions.
    """
    occupancy = {}
    for pred in predictions:
        slot_id = pred.get("class")
        # Assuming the model returns 'occupied' or 'vacant' as class
        occupancy[slot_id] = "occupied" if "occupied" in slot_id.lower() else "vacant"
    return occupancy