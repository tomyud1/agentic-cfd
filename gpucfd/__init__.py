# Keep this file free of imports: gpucfd.grids and gpucfd.bodies are plain NumPy and must import without
# the CUDA library (report scripts use them on machines with no GPU). gcfd.py loads the library on first use.
