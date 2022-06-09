import pytcherplants.color_analysis as ca

plant_path = 'samples/plants/1_14_19.10_30_20.p001.masked.jpg'


def test_analyze():
    df = ca.analyze_file(plant_path)
    print(df)
