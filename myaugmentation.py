import numpy as np

def flip_extend(X, y):
    self_flippable_horizontally = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
    self_flippable_vertically   = np.array([1, 5, 12, 15, 17])
    self_flippable_both         = np.array([32, 40])
    cross_flippable = np.array([
        [19, 20], [33, 34], [36, 37], [38, 39],
        [20, 19], [34, 33], [37, 36], [39, 38],
    ])
    num_classes = 43

    # Pre-compute masks for all classes once — O(n) instead of O(n * num_classes)
    class_masks = {c: (y == c) for c in range(num_classes)}
    cross_flip_map = {row[0]: row[1] for row in cross_flippable}

    X_parts = []
    y_parts = []

    for c in range(num_classes):
        #create a mask for the current class and select the corresponding samples
        mask = class_masks[c]
        X_c = X[mask]
        #start with the original samples for this class
        pieces = [X_c]
        #if this class is horizontally flippable, add the flipped samples
        if c in self_flippable_horizontally:
            pieces.append(X_c[:, :, ::-1, :])

        #if this class has a cross-flip mapping, add the flipped samples from the corresponding class
        if c in cross_flip_map:
            src = cross_flip_map[c]
            pieces.append(X[class_masks[src]][:, :, ::-1, :])

        #combine all pieces for this class into one array
        X_class = np.concatenate(pieces, axis=0)
        #add vertically flipped samples if applicable
        if c in self_flippable_vertically:
            X_class = np.concatenate([X_class, X_class[:, ::-1, :, :]], axis=0)
        #add both flipped samples if applicable
        if c in self_flippable_both:
            X_class = np.concatenate([X_class, X_class[:, ::-1, ::-1, :]], axis=0)
        #append the augmented samples and their corresponding labels to the parts lists
        X_parts.append(X_class)
        y_parts.append(np.full(X_class.shape[0], c, dtype=y.dtype))

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)