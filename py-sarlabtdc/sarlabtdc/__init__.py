import os as _os
import warnings as _warnings

# Note, if this is not here then Python doesn't want to find the CUDA DLL's
# even when they are on the path!
# Really not sure why... but oh well.
# Also this is only tested on Windows. Since DLL's are a Windows thing
# I assume that "os.add_dll_directory" will not work on Linux.
if "CUDA_PATH" not in _os.environ:
    _warnings.warn(
        "pyseymour may not work if the CUDA_PATH environment variable is not set"
    )
else:
    cuda_path = _os.environ["CUDA_PATH"]

    # Last component of the path is the version string, e.g. v13.2
    # Strip out the first character ('v') and then split to get the major version
    path_components = _os.path.split(cuda_path)
    major_version = int(path_components[-1][1:].split(".")[0])
    if major_version == 12:
        _os.add_dll_directory(_os.path.join(cuda_path, "bin"))
    elif major_version == 13:
        # Nvidia changed the location of the Cuda DLLs in version 13
        _os.add_dll_directory(_os.path.join(cuda_path, "bin", "x64"))
    else:
        raise ImportError(
            f"Unknown version of Cuda ({major_version}). You need to check where "
            "the Cuda DLLs are found and add a new elif branch to the pyseymour __init__.py."
        )

from _sarlabtdc import *
