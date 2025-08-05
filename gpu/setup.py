from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="bbox_cuda",
    ext_modules=[
        CUDAExtension(
            "bbox_cuda",
            ["bbox_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
