from pathlib import Path
from os.path import join
from os import getcwd

for path in Path('.').rglob('*.JPG'):
	new_name = join(getcwd(), f"{str(path.parents[0]).replace('/', '.').replace('-', '_')}.{path.name}")
	print(f"Moving {path.name} to {new_name}")
	path.rename(Path(new_name))
