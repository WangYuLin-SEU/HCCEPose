"""Read the camera and select object IDs for the RGB and RGB-D examples."""
import json
from numbers import Integral

import numpy as np


def read_intrinsics(path):
    with open(path, encoding='utf-8') as handle:
        camera = np.asarray(json.load(handle), dtype=np.float32)
    if camera.shape != (3, 3) or not np.isfinite(camera).all() or camera[0, 0] <= 0 or camera[1, 1] <= 0:
        raise ValueError('Intrinsics must be a finite 3 x 3 JSON array with positive focal lengths')
    return camera


def checked_object_ids(object_ids, available_ids):
    """Preserve request order and reject IDs that cannot identify a model unambiguously."""
    available = set(available_ids)
    seen = set()
    requested = []
    for obj_id in object_ids:
        if isinstance(obj_id, bool) or not isinstance(obj_id, Integral):
            raise ValueError(f'Expected an integer object ID, received {obj_id!r}')
        if obj_id not in available:
            raise ValueError(f'Unknown object ID {obj_id} in this dataset')
        if obj_id in seen:
            raise ValueError(f'Duplicate object ID {obj_id} in the request')
        seen.add(obj_id)
        requested.append(int(obj_id))
    return requested
