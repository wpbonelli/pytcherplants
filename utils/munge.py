from pathlib import Path
from os.path import join
from os import getcwd, listdir
from shutil import copyfile
from pprint import pprint


def flatten_nested_folders():
	for path in Path('.').rglob('*.JPG'):
		new_name = join(getcwd(), f"{str(path.parents[0]).replace('/', '.').replace('-', '_')}.{path.name}")
		print(f"Moving {path.name} to {new_name}")
		path.rename(Path(new_name))


def munge_plants(path: str):
	image_paths = list(Path(path).rglob('*.jpg'))
	print(f"Found paths:")
	pprint(image_paths)
	files = [p for p in image_paths if Path(p).is_file()]
	print(f"Found files:")
	pprint(files)
	for f in files:
		stem = f.stem
		fixed_stem = stem.replace('Maxsea', 'MaxSea').replace('Calmag', 'CalMag')
		print(f"Renaming {stem} to {fixed_stem}")
		f.rename(f"{fixed_stem}.jpg")


def reformat_dates(input_path: str, output_path: str):
	if not Path(input_path).is_dir(): raise ValueError(f"Invalid input directory: {output_path}")
	if not Path(output_path).is_dir(): raise ValueError(f"Invalid output directory: {output_path}")

	image_paths = list(Path(input_path).rglob('*.jpg'))
	print(f"Found paths:")
	pprint(image_paths)
	files = [p for p in image_paths if Path(p).is_file()]
	print(f"Found files:")
	pprint(files)
	for f in files:
		stem = f.stem
		spl = stem.split('.')
		if len(spl) < 3: continue
		date = spl[0]
		treatment = spl[1]
		name = spl[2]
		dspl = date.split('_')
		if len(dspl) != 3:
			print(f"Invalid date format: {date}")
			continue
		month = dspl[0]
		day = dspl[1]
		year = dspl[2]
		fixed_stem = f"{year}_{month}_{day}.{treatment}.{name}"
		input_file = join(input_path, f.name)
		output_file = join(output_path, f"{fixed_stem}.jpg")
		print(f"Copying {stem} to {output_file}")
		copyfile(input_file, output_file)
		# f.rename(f"{fixed_stem}.jpg")


