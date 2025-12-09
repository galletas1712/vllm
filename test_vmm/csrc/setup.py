"""Build script for shareable_cumem C extension."""

from setuptools import setup, Extension
import subprocess
import os


def get_cuda_path():
    """Find CUDA installation path."""
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        return cuda_home

    try:
        nvcc = subprocess.check_output(['which', 'nvcc']).decode().strip()
        return os.path.dirname(os.path.dirname(nvcc))
    except:
        pass

    for path in ['/usr/local/cuda', '/usr/cuda']:
        if os.path.exists(path):
            return path

    raise RuntimeError("CUDA not found. Set CUDA_HOME environment variable.")


cuda_path = get_cuda_path()
print(f"Using CUDA from: {cuda_path}")

ext = Extension(
    '_shareable_cumem_ext',
    sources=['shareable_cumem.cpp'],
    include_dirs=[
        os.path.join(cuda_path, 'include'),
    ],
    library_dirs=[
        os.path.join(cuda_path, 'lib64'),
        os.path.join(cuda_path, 'lib'),
    ],
    libraries=['cuda'],
    extra_compile_args=['-std=c++17', '-O3'],
    language='c++',
)

setup(
    name='shareable_cumem',
    version='0.1',
    description='Shareable CUDA memory allocator using VMM',
    ext_modules=[ext],
)
