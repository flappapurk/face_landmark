import numpy as np
import requests


def download_file(url, save_path):
    print(f'downloading from {url} to {save_path}')
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
            print("file downloaded successfully!")
    else:
        print("Failed to download file")


def face_coverage(x):
    xp = [0, 1]
    fp = [0.0, 0.1685]
    return np.interp(x, xp, fp)


def polygon_area(coordinates):
    n = len(coordinates)
    area = 0

    for i in range(n):
        x1, y1 = coordinates[i]
        # Wrap around to the first vertex for the last edge
        x2, y2 = coordinates[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)

    return abs(area) / 2
