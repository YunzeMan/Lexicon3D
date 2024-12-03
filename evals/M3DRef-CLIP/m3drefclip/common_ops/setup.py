from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='COMMON_OPS',
    version="1.0",
    ext_modules=[
        CUDAExtension('COMMON_OPS', [
            'src/common_ops_api.cpp',
            'src/common_ops.cpp',
            'src/cuda.cu'
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
