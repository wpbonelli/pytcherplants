from pathlib import Path
from os.path import join
from os import getcwd, listdir
from shutil import copyfile


def flatten_nested_folders():
	for path in Path('.').rglob('*.JPG'):
		new_name = join(getcwd(), f"{str(path.parents[0]).replace('/', '.').replace('-', '_')}.{path.name}")
		print(f"Moving {path.name} to {new_name}")
		path.rename(Path(new_name))


def munge_plants():
	paths = Path('.').rglob('*.jpg')
	files = [path for path in paths if Path(path).is_file()]
	for path in files:
		stem = path.stem
		fixed_stem = stem.replace('Maxsea', 'MaxSea').replace('Calmag', 'CalMag')
		path.rename(f"{fixed_stem}.jpg")



