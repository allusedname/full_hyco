import os
from abc import ABC


class ImagePathToInformationMapping(ABC):
    def __init__(self):
        pass

    def __call__(self, full_path):
        raise NotImplementedError()


class ImageNetInfoMapping(ImagePathToInformationMapping):
    """
        For ImageNet-like directory structures without sessions/conditions:
        .../{category}/{img_name}
    """

    def __call__(self, full_path):
        session_name = "session-1"
        img_name = os.path.basename(full_path)
        condition = "NaN"
        category = os.path.basename(os.path.dirname(full_path))
        return session_name, img_name, condition, category


class ImageNetCInfoMapping(ImagePathToInformationMapping):
    """
        For the ImageNet-C Dataset with path structure:
        .../{corruption function}/{corruption severity}/{category}/{img_name}
    """

    def __call__(self, full_path):
        session_name = "session-1"
        img_name = os.path.basename(full_path)
        level1 = os.path.dirname(full_path)          # .../{category}
        level2 = os.path.dirname(level1)             # .../{severity}/{category}
        level3 = os.path.dirname(level2)             # .../{corruption}/{severity}/{category}
        category = os.path.basename(level1)
        severity = os.path.basename(level2)
        corruption = os.path.basename(level3)
        condition = f"{corruption}-{severity}"
        return session_name, img_name, condition, category


class InfoMappingWithSessions(ImagePathToInformationMapping):
    """
        Directory/filename structure:
        .../{session_name}/{something}_{something}_{something}_{condition}_{category}_{img_name}
    """

    def __call__(self, full_path):
        session_name = os.path.basename(os.path.dirname(full_path))
        img_name = os.path.basename(full_path)
        parts = img_name.split("_")
        # assuming parts[3] is condition and parts[4] is category
        condition = parts[3] if len(parts) > 3 else ""
        category = parts[4] if len(parts) > 4 else ""
        return session_name, img_name, condition, category
