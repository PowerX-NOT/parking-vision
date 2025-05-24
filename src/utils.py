import os

def get_file_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        return 'image'
    elif ext in ['.mp4', '.avi']:
        return 'video'
    else:
        return 'unknown'