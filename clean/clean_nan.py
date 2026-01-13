import math

def clean_nan(value):
    """
    Chuyển NaN (float) sang None (null trong JSON)
    """
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def clean_nan_object(obj):
    """
    Làm sạch NaN cho:
    - dict
    - list
    - value đơn lẻ
    (đệ quy)
    """
    if isinstance(obj, dict):
        return {
            key: clean_nan_object(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            clean_nan_object(item)
            for item in obj
        ]

    return clean_nan(obj)
