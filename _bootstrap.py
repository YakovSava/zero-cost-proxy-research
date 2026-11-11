import pathlib
import sys

ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent
COMFY_ROOT: pathlib.Path = ROOT.parents[1]

if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))
