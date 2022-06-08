import cv2

from pytcherplants.clustering import get_clusters

input_path = 'samples/masks/1_14_19.10_30_20.5V4B3121.masked.jpg'


def test_get_clusters_default_k_and_no_filters():
    image = cv2.imread(input_path)
    counts = get_clusters(image)
    assert len(counts.keys()) == 14
    assert all(v > 0 for v in counts.values())
    print(counts)


def test_get_clusters_default_k_and_filters():
    image = cv2.imread(input_path)
    counts = get_clusters(image, filters=[((0, 38, 40), (39, 255, 255))])  # filter out blues/purples
    assert len(counts.keys()) == 14
    assert all(v > 0 for v in counts.values())
    print(counts)


def test_get_clusters_custom_k():
    image = cv2.imread(input_path)
    counts = get_clusters(image, k=10)
    assert len(counts.keys()) == 9
    assert all(v > 0 for v in counts.values())
    print(counts)
