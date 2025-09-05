load("//bzl/tools/python:defs.bzl", "py_binary", "py_library")

py_library(
    name = "lwabudget_lib",
    srcs = ["lwabudget.py"],
    deps = [
        "@pypi//numpy",
    ],
)

py_binary(
    name = "lwa_budget_make",
    srcs = ["LWA_budget_make.py"],
    deps = [
        ":lwabudget_lib",
        "@pypi//numpy",
        "@pypi//netCDF4",
    ],
)

py_binary(
    name = "lp_budget_make",
    srcs = ["LP_budget_make.py"],
    deps = [
        ":lwabudget_lib",
        "@pypi//numpy",
        "@pypi//netCDF4",
    ],
) 