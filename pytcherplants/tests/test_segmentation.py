import pytcherplants.segmentation as seg

plant_path = 'samples/plants/1_14_19.10_30_20.p001.masked.jpg'
group_path = 'samples/groups/1_14_19.10_30_20.5V4B3121.masked.jpg'


def test_segment_plant():
    plants, labelled = seg.segment_plants(plant_path)


def test_segment_group():
    plants, labelled = seg.segment_plants(group_path, 6)

