def split_into_segments(points):
    """
    Splits a sorted list of points into continuous segments.

    Parameters
    ----------
    points : list of int
        A sorted list of integer points.

    Returns
    -------
    list of list of int
        A list of continuous segments, where each segment is a list of integers.
    """
    segments = []
    current_segment = [points[0]]
    for i in range(1, len(points)):
        if points[i] == points[i - 1] + 1:
            current_segment.append(points[i])
        else:
            segments.append(current_segment)
            current_segment = [points[i]]
    segments.append(current_segment)
    return segments
