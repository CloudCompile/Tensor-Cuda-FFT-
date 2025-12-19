import pathlib

p = pathlib.Path(__file__).resolve().parents[1]
for x in sorted(p.glob('*.py'), key=lambda y: y.name.lower()):
    print(x.name)
