import os
import warnings

# Note, if this is not here then Python doesn't want to find the CUDA DLL's
# even when they are on the path!
# Really not sure why... but oh well.
# Also this is only tested on Windows. Since DLL's are a Windows thing
# I assume that "os.add_dll_directory" will not work on Linux.
if "CUDA_PATH" not in os.environ:
    warnings.warn(
        "sarlabtdc might not work if the CUDA_PATH environment variable does not exist"
    )
else:
    cuda_path = os.environ["CUDA_PATH"]
    os.add_dll_directory(os.path.join(cuda_path, "bin"))

from _sarlabtdc import *
