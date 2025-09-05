load("//bzl/tools/python:defs.bzl", "py_binary", "py_library")

py_library(
    name = "dry_matrices",
    srcs = [
        "build_matrices.py",
    ],
    deps = [
        "@pypi//numpy",
    ],
)

py_binary(
    name = "calculate_eddy_growth",
    srcs = ["Calculate_Eddy_Growth.py"],
    deps = [
        ":dry_matrices",
        "@pypi//numpy",
        "@pypi//scipy",
        "@pypi//netCDF4",
    ],
) 