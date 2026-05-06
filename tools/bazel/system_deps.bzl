# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local system dependency repositories for TensorRT-LLM native Bazel builds."""

_CUDA_ENV_VARS = ["CUDA_HOME", "CUDA_PATH"]
_TENSORRT_ENV_VARS = ["TensorRT_ROOT", "TENSORRT_ROOT"]
_NCCL_ENV_VARS = ["NCCL_ROOT"]
_MPI_ENV_VARS = ["MPI_HOME", "MPI_ROOT"]
_TORCH_ENV_VARS = ["TORCH_INSTALL_PREFIX", "TORCH_ROOT"]
_NANOBIND_ROOT_ENV_VARS = ["NANOBIND_ROOT"]
_NANOBIND_INCLUDE_ENV_VARS = ["NANOBIND_INCLUDE_DIR"]
_NLOHMANN_JSON_ENV_VARS = ["NLOHMANN_JSON_ROOT"]
_CUTLASS_ENV_VARS = ["CUTLASS_ROOT"]
_SYSTEM_DEP_ENV_VARS = (
    _CUDA_ENV_VARS +
    _TENSORRT_ENV_VARS +
    _NCCL_ENV_VARS +
    _MPI_ENV_VARS +
    _TORCH_ENV_VARS +
    _NANOBIND_ROOT_ENV_VARS +
    _NANOBIND_INCLUDE_ENV_VARS +
    _NLOHMANN_JSON_ENV_VARS +
    _CUTLASS_ENV_VARS
)


def _clean_path(path):
    if len(path) > 1 and path.endswith("/"):
        return path[:-1]
    return path


def _join(root, rel):
    root = _clean_path(root)
    if rel:
        return root + "/" + rel
    return root


def _first_env(repository_ctx, env_vars):
    for env_var in env_vars:
        value = repository_ctx.os.environ.get(env_var)
        if value:
            return env_var, _clean_path(value)
    return None, None


def _header_names(header_candidates):
    return ", ".join([candidate[1] for candidate in header_candidates])


def _header_path(include_dir, header):
    if include_dir:
        return include_dir + "/" + header
    return header


def _header_paths(header_candidates):
    return ", ".join([_header_path(candidate[0], candidate[1]) for candidate in header_candidates])


def _find_header_dir(repository_ctx, root, header_candidates):
    for include_dir, header in header_candidates:
        header_path = _join(root, _header_path(include_dir, header))
        if repository_ctx.path(header_path).exists:
            return _join(root, include_dir)
    return None


def _required_header_dirs(repository_ctx, root, header_candidates):
    include_dirs = []
    missing_headers = []
    for include_dir, header in header_candidates:
        header_path = _join(root, _header_path(include_dir, header))
        if repository_ctx.path(header_path).exists:
            full_include_dir = _join(root, include_dir)
            if full_include_dir not in include_dirs:
                include_dirs.append(full_include_dir)
        else:
            missing_headers.append(_header_path(include_dir, header))
    return include_dirs, missing_headers


def _select_root_with_required_headers(repository_ctx, dep_name, env_vars, default_roots, header_candidates):
    env_var, env_root = _first_env(repository_ctx, env_vars)
    if env_root:
        include_dirs, missing_headers = _required_header_dirs(repository_ctx, env_root, header_candidates)
        if not missing_headers:
            return env_root, include_dirs
        fail(
            (
                "%s repository %s=%s is missing required headers %s. " +
                "Set %s to a dependency root containing %s."
            ) % (dep_name, env_var, env_root, ", ".join(missing_headers), env_var, _header_paths(header_candidates))
        )

    for root in default_roots:
        include_dirs, missing_headers = _required_header_dirs(repository_ctx, root, header_candidates)
        if not missing_headers:
            return _clean_path(root), include_dirs

    fail(
        "%s repository could not find required headers %s. Set one of %s. Tried default roots: %s." %
        (dep_name, _header_paths(header_candidates), ", ".join(env_vars), ", ".join(default_roots))
    )


def _select_root(repository_ctx, dep_name, env_vars, default_roots, header_candidates):
    env_var, env_root = _first_env(repository_ctx, env_vars)
    if env_root:
        include_dir = _find_header_dir(repository_ctx, env_root, header_candidates)
        if include_dir:
            return env_root, include_dir
        fail(
            (
                "%s repository %s=%s does not contain required header %s. " +
                "Use an absolute %s path with an include directory for this dependency."
            ) % (dep_name, env_var, env_root, _header_names(header_candidates), env_var)
        )

    for root in default_roots:
        include_dir = _find_header_dir(repository_ctx, root, header_candidates)
        if include_dir:
            return _clean_path(root), include_dir

    fail(
        "%s repository could not find required header %s. Set one of %s. Tried default roots: %s." %
        (dep_name, _header_names(header_candidates), ", ".join(env_vars), ", ".join(default_roots))
    )


def _workspace_root(repository_ctx):
    return str(repository_ctx.path(repository_ctx.attr._workspace_file).dirname)


def _workspace_default_roots(repository_ctx, rel_roots):
    workspace_root = _workspace_root(repository_ctx)
    return [_join(workspace_root, rel_root) for rel_root in rel_roots]


def _require_file_contains(repository_ctx, dep_name, base_dir, rel_path, required_snippets, env_vars):
    file_path = _join(base_dir, rel_path)
    if not repository_ctx.path(file_path).exists:
        fail(
            "%s repository found root %s but could not find %s. Set one of %s to a complete dependency root." %
            (dep_name, base_dir, rel_path, ", ".join(env_vars))
        )
    content = repository_ctx.read(file_path)
    missing_snippets = [snippet for snippet in required_snippets if content.find(snippet) < 0]
    if missing_snippets:
        fail(
            (
                "%s repository found %s but it does not contain expected markers %s. " +
                "Set one of %s to a dependency root matching the version used by TensorRT-LLM."
            ) % (dep_name, file_path, ", ".join(missing_snippets), ", ".join(env_vars))
        )


def _strip_wrapping_parens(token):
    token = token.strip()
    for _ in range(len(token)):
        if not token.startswith("(") or not token.endswith(")"):
            break
        token = token[1:-1].strip()
    return token


def _parse_int_literal(token):
    token = _strip_wrapping_parens(token)
    for suffix in ["ULL", "ull", "UL", "ul", "LL", "ll", "U", "u", "L", "l"]:
        if token.endswith(suffix):
            token = token[:-len(suffix)]
            break
    if not token:
        return None
    for index in range(len(token)):
        if "0123456789".find(token[index]) < 0:
            return None
    return int(token)


def _split_words(line):
    words = []
    for word in line.strip().replace("\t", " ").split(" "):
        if word:
            words.append(word)
    return words


def _is_comment_token(token):
    return token.startswith("//") or token.startswith("/*")


def _simple_macro_definition(tokens):
    if len(tokens) < 3 or tokens[0] != "#define" or tokens[1].find("(") >= 0:
        return None
    if len(tokens) > 3 and not _is_comment_token(tokens[3]):
        return None
    return tokens[1], tokens[2]


def _read_simple_macro_definitions(content):
    definitions = {}
    for line in content.splitlines():
        definition = _simple_macro_definition(_split_words(line))
        if definition:
            macro_name, value_token = definition
            if macro_name not in definitions:
                definitions[macro_name] = []
            definitions[macro_name].append(value_token)
    return definitions


def _resolve_macro_int(definitions, macro_name):
    pending = [macro_name]
    queued = {macro_name: True}
    seen = {}
    for index in range(len(definitions) + 1):
        if index >= len(pending):
            return None
        current_macro = pending[index]
        if current_macro in seen:
            continue
        seen[current_macro] = True
        for value_token in definitions.get(current_macro, []):
            value = _parse_int_literal(value_token)
            if value != None:
                return value
            alias_token = _strip_wrapping_parens(value_token)
            if alias_token in definitions and alias_token not in queued:
                pending.append(alias_token)
                queued[alias_token] = True
    return None


def _format_macro_definition(macro_name, definitions):
    if macro_name not in definitions:
        return macro_name + " is not defined"
    return macro_name + " is defined as " + ", ".join(definitions[macro_name])


def _read_macro_ints(repository_ctx, dep_name, header_path, macro_names, env_vars):
    content = repository_ctx.read(header_path)
    definitions = _read_simple_macro_definitions(content)
    values = {}
    for macro_name in macro_names:
        value = _resolve_macro_int(definitions, macro_name)
        if value != None:
            values[macro_name] = value

    missing_macros = [macro_name for macro_name in macro_names if macro_name not in values]
    if missing_macros:
        fail(
            (
                "%s repository found %s but no integer value was available for required version macros %s. " +
                "Simple aliases in the same header are resolved; unsupported definitions were: %s. " +
                "Set one of %s to a complete dependency root."
            ) % (
                dep_name,
                header_path,
                ", ".join(missing_macros),
                "; ".join([_format_macro_definition(macro_name, definitions) for macro_name in missing_macros]),
                ", ".join(env_vars),
            )
        )
    return values


def _version_less(version, minimum):
    for index in range(len(minimum)):
        if version[index] < minimum[index]:
            return True
        if version[index] > minimum[index]:
            return False
    return False


def _format_version(version):
    return ".".join([str(part) for part in version])


def _require_minimum_macro_version(repository_ctx, dep_name, header_path, macro_names, minimum, env_vars):
    values = _read_macro_ints(repository_ctx, dep_name, header_path, macro_names, env_vars)
    version = [values[macro_name] for macro_name in macro_names]
    if _version_less(version, minimum):
        fail(
            (
                "%s repository found version %s in %s, but TensorRT-LLM requires at least %s. " +
                "Set one of %s to a compatible dependency root."
            ) % (dep_name, _format_version(version), header_path, _format_version(minimum), ", ".join(env_vars))
        )


def _format_cuda_version(cuda_version):
    return "%d.%d" % (cuda_version // 1000, (cuda_version % 1000) // 10)


def _require_minimum_cuda_version(repository_ctx, header_path, minimum):
    values = _read_macro_ints(repository_ctx, "CUDA", header_path, ["CUDA_VERSION"], _CUDA_ENV_VARS)
    cuda_version = values["CUDA_VERSION"]
    if cuda_version < minimum:
        fail(
            (
                "CUDA repository found CUDA_VERSION %s (%s) in %s, but TensorRT-LLM requires at least %s (%s). " +
                "Set one of %s to a compatible CUDA Toolkit root."
            ) % (
                cuda_version,
                _format_cuda_version(cuda_version),
                header_path,
                minimum,
                _format_cuda_version(minimum),
                ", ".join(_CUDA_ENV_VARS),
            )
        )


def _required_existing_file(repository_ctx, dep_name, paths, description, env_vars):
    for path in paths:
        if repository_ctx.path(path).exists:
            return path
    fail(
        "%s repository could not find %s. Set one of %s to a complete dependency root." %
        (dep_name, description, ", ".join(env_vars))
    )


def _lib_candidate_paths(root, lib_names, rel_lib_dirs, abs_lib_dirs):
    paths = []
    for rel_lib_dir in rel_lib_dirs:
        for lib_name in lib_names:
            paths.append(_join(root, rel_lib_dir + "/" + lib_name))
    for abs_lib_dir in abs_lib_dirs:
        for lib_name in lib_names:
            paths.append(_join(abs_lib_dir, lib_name))
    return paths


def _find_lib(repository_ctx, root, lib_names, rel_lib_dirs, abs_lib_dirs):
    for path in _lib_candidate_paths(root, lib_names, rel_lib_dirs, abs_lib_dirs):
        if repository_ctx.path(path).exists:
            return path
    return None


def _required_lib(repository_ctx, dep_name, target_name, root, lib_names, rel_lib_dirs, abs_lib_dirs, env_vars):
    path = _find_lib(repository_ctx, root, lib_names, rel_lib_dirs, abs_lib_dirs)
    if path:
        return path
    fail(
        (
            "%s repository found root %s but could not find %s library (%s). " +
            "Set one of %s to the dependency root that contains this library."
        ) % (dep_name, root, target_name, ", ".join(lib_names), ", ".join(env_vars))
    )


def _optional_lib(repository_ctx, root, lib_names, rel_lib_dirs, abs_lib_dirs):
    return _find_lib(repository_ctx, root, lib_names, rel_lib_dirs, abs_lib_dirs)


def _lib_spec(name, source_path, link_name, deps = None):
    if deps == None:
        deps = []
    return {
        "deps": deps,
        "dest": "lib/" + link_name,
        "name": name,
        "source": source_path,
    }


def _include_dir_link(include_dir):
    return [{
        "dest": "include",
        "source": include_dir,
    }]


def _header_file_link(include_dir, header):
    return [{
        "dest": "include/" + header,
        "source": _join(include_dir, header),
    }]


def _format_list(items):
    if not items:
        return "[]"
    return "[\n" + "".join(["        \"%s\",\n" % item for item in items]) + "    ]"


def _render_build_file(libraries, aggregate_name, include_paths = None, aliases = None):
    if include_paths == None:
        include_paths = ["include"]
    if aliases == None:
        aliases = []

    library_files = []
    for library in libraries:
        library_files.extend(library["runfiles"])
    lines = [
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "# Generated by //tools/bazel:system_deps.bzl.",
        "",
        "package(default_visibility = [\"//visibility:public\"])",
        "",
        "filegroup(",
        "    name = \"header_files\",",
        "    srcs = glob(",
        "        [\"include/**\"],",
        "        exclude = [\"include/.empty\"],",
        "    ),",
        ")",
        "",
        "cc_library(",
        "    name = \"headers\",",
        "    hdrs = [\":header_files\"],",
        "    includes = %s," % _format_list(include_paths),
        ")",
        "",
        "filegroup(",
        "    name = \"shared_libraries\",",
        "    srcs = %s," % _format_list(library_files),
        ")",
        "",
    ]

    for library in libraries:
        import_name = "_" + library["name"] + "_import"
        deps = [":headers", ":" + import_name] + [":" + dep for dep in library["deps"]]
        lines.extend([
            "cc_import(",
            "    name = \"%s\"," % import_name,
            "    shared_library = \"%s\"," % library["dest"],
            "    visibility = [\"//visibility:private\"],",
            ")",
            "",
            "cc_library(",
            "    name = \"%s\"," % library["name"],
            "    data = %s," % _format_list(library["runfiles"]),
            "    deps = %s," % _format_list(deps),
            ")",
            "",
        ])

    library_names = [library["name"] for library in libraries]
    if aggregate_name not in library_names:
        lines.extend([
            "cc_library(",
            "    name = \"%s\"," % aggregate_name,
            "    deps = %s," % _format_list([":" + library for library in library_names]),
            ")",
            "",
        ])

    for alias in aliases:
        lines.extend([
            "alias(",
            "    name = \"%s\"," % alias["name"],
            "    actual = \":%s\"," % alias["actual"],
            ")",
            "",
        ])
    return "\n".join(lines)


def _render_header_build_file(primary_name, header_filegroup, header_globs, include_paths, aliases = None, duplicate_names = None):
    if aliases == None:
        aliases = []
    if duplicate_names == None:
        duplicate_names = []

    lines = [
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "# SPDX-License-Identifier: Apache-2.0",
        "# Generated by //tools/bazel:system_deps.bzl.",
        "",
        "package(default_visibility = [\"//visibility:public\"])",
        "",
        "filegroup(",
        "    name = \"%s\"," % header_filegroup,
        "    srcs = glob(%s)," % _format_list(header_globs),
        ")",
        "",
        "cc_library(",
        "    name = \"%s\"," % primary_name,
        "    hdrs = [\":%s\"]," % header_filegroup,
        "    includes = %s," % _format_list(include_paths),
        ")",
        "",
    ]

    for duplicate_name in duplicate_names:
        lines.extend([
            "cc_library(",
            "    name = \"%s\"," % duplicate_name,
            "    hdrs = [\":%s\"]," % header_filegroup,
            "    includes = %s," % _format_list(include_paths),
            ")",
            "",
        ])

    for alias_name in aliases:
        lines.extend([
            "alias(",
            "    name = \"%s\"," % alias_name,
            "    actual = \":%s\"," % primary_name,
            ")",
            "",
        ])

    return "\n".join(lines)


def _write_symlinked_repo(repository_ctx, links, build_file_content):
    for link in links:
        repository_ctx.symlink(repository_ctx.path(link["source"]), link["dest"])
    repository_ctx.file("BUILD.bazel", build_file_content)


def _with_library_runfiles(repository_ctx, libraries):
    materialized_libraries = []
    for library in libraries:
        source = repository_ctx.path(library["source"])
        real_source = source.realpath
        link_name = library["dest"].split("/")[-1]
        runfiles = [library["dest"]]
        links = [{
            "dest": library["dest"],
            "source": str(source),
        }]

        # Keep the unversioned link name for linking and add the real SONAME
        # basename to runfiles when the selected library resolves through a
        # versioned shared-library symlink.
        if real_source.basename != link_name:
            real_dest = "lib/" + real_source.basename
            runfiles.append(real_dest)
            links.append({
                "dest": real_dest,
                "source": str(real_source),
            })

        materialized_libraries.append({
            "deps": library["deps"],
            "dest": library["dest"],
            "links": links,
            "name": library["name"],
            "runfiles": runfiles,
            "source": library["source"],
        })
    return materialized_libraries


def _write_system_repo(repository_ctx, include_links, libraries, aggregate_name, include_paths = None, aliases = None):
    libraries = _with_library_runfiles(repository_ctx, libraries)
    for include_link in include_links:
        if include_link["dest"].startswith("include/"):
            repository_ctx.file("include/.empty", "")
        repository_ctx.symlink(repository_ctx.path(include_link["source"]), include_link["dest"])
    repository_ctx.file("lib/.empty", "")
    for library in libraries:
        for link in library["links"]:
            repository_ctx.symlink(repository_ctx.path(link["source"]), link["dest"])
    repository_ctx.file("BUILD.bazel", _render_build_file(libraries, aggregate_name, include_paths, aliases))


def _cuda_repository_impl(repository_ctx):
    root, include_dir = _select_root(
        repository_ctx,
        "CUDA",
        _CUDA_ENV_VARS,
        ["/usr/local/cuda"],
        [
            ("include", "cuda.h"),
            ("targets/x86_64-linux/include", "cuda.h"),
        ],
    )
    _require_minimum_cuda_version(repository_ctx, _join(include_dir, "cuda.h"), 11020)
    cuda_lib_dirs = [
        "lib64",
        "lib",
        "targets/x86_64-linux/lib",
        "compat/lib.real",
        "targets/x86_64-linux/lib/stubs",
    ]
    cuda_driver_lib_dirs = ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"]
    libraries = [
        _lib_spec(
            "cuda_driver",
            _required_lib(
                repository_ctx,
                "CUDA",
                "cuda_driver",
                root,
                ["libcuda.so", "libcuda.so.1"],
                cuda_lib_dirs,
                cuda_driver_lib_dirs,
                _CUDA_ENV_VARS,
            ),
            "libcuda.so",
        ),
        _lib_spec(
            "cudart",
            _required_lib(
                repository_ctx,
                "CUDA",
                "cudart",
                root,
                ["libcudart.so"],
                cuda_lib_dirs,
                [],
                _CUDA_ENV_VARS,
            ),
            "libcudart.so",
        ),
        _lib_spec(
            "cublas",
            _required_lib(
                repository_ctx,
                "CUDA",
                "cublas",
                root,
                ["libcublas.so"],
                cuda_lib_dirs,
                [],
                _CUDA_ENV_VARS,
            ),
            "libcublas.so",
        ),
        _lib_spec(
            "cublasLt",
            _required_lib(
                repository_ctx,
                "CUDA",
                "cublasLt",
                root,
                ["libcublasLt.so"],
                cuda_lib_dirs,
                [],
                _CUDA_ENV_VARS,
            ),
            "libcublasLt.so",
            deps = ["cublas"],
        ),
        _lib_spec(
            "nvrtc",
            _required_lib(
                repository_ctx,
                "CUDA",
                "nvrtc",
                root,
                ["libnvrtc.so"],
                cuda_lib_dirs,
                [],
                _CUDA_ENV_VARS,
            ),
            "libnvrtc.so",
        ),
    ]
    nvml = _optional_lib(
        repository_ctx,
        root,
        ["libnvidia-ml.so", "libnvidia-ml.so.1"],
        cuda_lib_dirs,
        cuda_driver_lib_dirs,
    )
    if nvml:
        libraries.append(_lib_spec("nvml", nvml, "libnvidia-ml.so"))

    _write_system_repo(
        repository_ctx,
        _include_dir_link(include_dir),
        libraries,
        "cuda",
        aliases = [
            {
                "actual": "headers",
                "name": "cuda_headers",
            },
            {
                "actual": "cudart",
                "name": "cuda_runtime",
            },
        ],
    )


def _tensorrt_repository_impl(repository_ctx):
    root, include_dir = _select_root(
        repository_ctx,
        "TensorRT",
        _TENSORRT_ENV_VARS,
        ["/usr/local/tensorrt"],
        [("include", "NvInfer.h")],
    )
    version_header = _required_existing_file(
        repository_ctx,
        "TensorRT",
        [
            _join(include_dir, "NvInferVersion.h"),
            _join(include_dir, "NvInferRuntimeBase.h"),
        ],
        "NvInferVersion.h or NvInferRuntimeBase.h",
        _TENSORRT_ENV_VARS,
    )
    _require_minimum_macro_version(
        repository_ctx,
        "TensorRT",
        version_header,
        ["NV_TENSORRT_MAJOR"],
        [10],
        _TENSORRT_ENV_VARS,
    )
    lib_dirs = ["lib", "lib64", "lib/stubs"]
    libraries = [
        _lib_spec(
            "nvinfer",
            _required_lib(
                repository_ctx,
                "TensorRT",
                "nvinfer",
                root,
                ["libnvinfer.so"],
                lib_dirs,
                [],
                _TENSORRT_ENV_VARS,
            ),
            "libnvinfer.so",
        ),
        _lib_spec(
            "nvonnxparser",
            _required_lib(
                repository_ctx,
                "TensorRT",
                "nvonnxparser",
                root,
                ["libnvonnxparser.so"],
                lib_dirs,
                [],
                _TENSORRT_ENV_VARS,
            ),
            "libnvonnxparser.so",
            deps = ["nvinfer"],
        ),
        _lib_spec(
            "nvinfer_plugin",
            _required_lib(
                repository_ctx,
                "TensorRT",
                "nvinfer_plugin",
                root,
                ["libnvinfer_plugin.so"],
                lib_dirs,
                [],
                _TENSORRT_ENV_VARS,
            ),
            "libnvinfer_plugin.so",
            deps = ["nvinfer"],
        ),
    ]
    _write_system_repo(repository_ctx, _include_dir_link(include_dir), libraries, "tensorrt")


def _nccl_repository_impl(repository_ctx):
    root, include_dir = _select_root(
        repository_ctx,
        "NCCL",
        _NCCL_ENV_VARS,
        ["/usr"],
        [
            ("include", "nccl.h"),
            ("include/x86_64-linux-gnu", "nccl.h"),
        ],
    )
    _require_minimum_macro_version(
        repository_ctx,
        "NCCL",
        _join(include_dir, "nccl.h"),
        ["NCCL_MAJOR", "NCCL_MINOR"],
        [2, 29],
        _NCCL_ENV_VARS,
    )
    lib_dirs = ["lib/x86_64-linux-gnu", "lib64", "lib"]
    libraries = [
        _lib_spec(
            "nccl",
            _required_lib(
                repository_ctx,
                "NCCL",
                "nccl",
                root,
                ["libnccl.so"],
                lib_dirs,
                [],
                _NCCL_ENV_VARS,
            ),
            "libnccl.so",
        ),
    ]
    _write_system_repo(repository_ctx, _header_file_link(include_dir, "nccl.h"), libraries, "nccl")


def _mpi_repository_impl(repository_ctx):
    root, include_dir = _select_root(
        repository_ctx,
        "MPI",
        _MPI_ENV_VARS,
        ["/usr/local/mpi", "/opt/hpcx/ompi"],
        [("include", "mpi.h")],
    )
    lib_dirs = ["lib", "lib64"]
    libraries = [
        _lib_spec(
            "mpi",
            _required_lib(
                repository_ctx,
                "MPI",
                "mpi",
                root,
                ["libmpi.so"],
                lib_dirs,
                [],
                _MPI_ENV_VARS,
            ),
            "libmpi.so",
        ),
    ]
    _write_system_repo(repository_ctx, _include_dir_link(include_dir), libraries, "mpi")


def _torch_repository_impl(repository_ctx):
    root, _ = _select_root_with_required_headers(
        repository_ctx,
        "Torch",
        _TORCH_ENV_VARS,
        ["/usr/local/lib/python3.12/dist-packages/torch"],
        [
            ("include", "torch/extension.h"),
            ("include", "ATen/ATen.h"),
            ("include", "c10/cuda/CUDAStream.h"),
            ("include/torch/csrc/api/include", "torch/torch.h"),
        ],
    )
    lib_dirs = ["lib"]
    libraries = [
        _lib_spec(
            "c10",
            _required_lib(
                repository_ctx,
                "Torch",
                "c10",
                root,
                ["libc10.so"],
                lib_dirs,
                [],
                _TORCH_ENV_VARS,
            ),
            "libc10.so",
        ),
        _lib_spec(
            "c10_cuda",
            _required_lib(
                repository_ctx,
                "Torch",
                "c10_cuda",
                root,
                ["libc10_cuda.so"],
                lib_dirs,
                [],
                _TORCH_ENV_VARS,
            ),
            "libc10_cuda.so",
            deps = ["c10"],
        ),
        _lib_spec(
            "torch_cpu",
            _required_lib(
                repository_ctx,
                "Torch",
                "torch_cpu",
                root,
                ["libtorch_cpu.so"],
                lib_dirs,
                [],
                _TORCH_ENV_VARS,
            ),
            "libtorch_cpu.so",
            deps = ["c10"],
        ),
        _lib_spec(
            "torch_cuda",
            _required_lib(
                repository_ctx,
                "Torch",
                "torch_cuda",
                root,
                ["libtorch_cuda.so"],
                lib_dirs,
                [],
                _TORCH_ENV_VARS,
            ),
            "libtorch_cuda.so",
            deps = ["c10", "c10_cuda", "torch_cpu"],
        ),
        _lib_spec(
            "torch",
            _required_lib(
                repository_ctx,
                "Torch",
                "torch",
                root,
                ["libtorch.so"],
                lib_dirs,
                [],
                _TORCH_ENV_VARS,
            ),
            "libtorch.so",
            deps = ["torch_cpu", "torch_cuda"],
        ),
        _lib_spec(
            "torch_python",
            _required_lib(
                repository_ctx,
                "Torch",
                "torch_python",
                root,
                ["libtorch_python.so"],
                lib_dirs,
                [],
                _TORCH_ENV_VARS,
            ),
            "libtorch_python.so",
            deps = ["torch"],
        ),
    ]
    _write_system_repo(
        repository_ctx,
        _include_dir_link(_join(root, "include")),
        libraries,
        "torch_libs",
        include_paths = ["include", "include/torch/csrc/api/include"],
    )


def _nanobind_repository_impl(repository_ctx):
    links = []
    include_paths = ["include"]
    header_globs = ["include/**"]
    include_env_var, include_dir = _first_env(repository_ctx, _NANOBIND_INCLUDE_ENV_VARS)

    if include_dir:
        header_path = _join(include_dir, "nanobind/nanobind.h")
        if not repository_ctx.path(header_path).exists:
            fail(
                (
                    "nanobind repository %s=%s does not contain nanobind/nanobind.h. " +
                    "Set %s to a nanobind include directory."
                ) % (include_env_var, include_dir, include_env_var)
            )
        links.append({"dest": "include", "source": include_dir})
    else:
        root, include_dirs = _select_root_with_required_headers(
            repository_ctx,
            "nanobind",
            _NANOBIND_ROOT_ENV_VARS,
            ["/usr/local/lib/python3.12/dist-packages/nanobind"] +
            _workspace_default_roots(repository_ctx, ["cpp/build/_deps/nanobind-src"]),
            [("include", "nanobind/nanobind.h")],
        )
        links.append({"dest": "include", "source": include_dirs[0]})

        robin_map_header = _join(root, "ext/robin_map/include/tsl/robin_map.h")
        if repository_ctx.path(robin_map_header).exists:
            links.append({"dest": "ext", "source": _join(root, "ext")})
            include_paths.append("ext/robin_map/include")
            header_globs.append("ext/robin_map/include/**")

        cmake_dir = _join(root, "cmake")
        if repository_ctx.path(_join(cmake_dir, "nanobind-config.cmake")).exists:
            links.append({"dest": "cmake", "source": cmake_dir})

    build_file = _render_header_build_file(
        "nanobind_headers",
        "nanobind_header_files",
        header_globs,
        include_paths,
        aliases = ["headers"],
    )
    _write_symlinked_repo(repository_ctx, links, build_file)


def _nlohmann_json_repository_impl(repository_ctx):
    _, include_dirs = _select_root_with_required_headers(
        repository_ctx,
        "nlohmann/json",
        _NLOHMANN_JSON_ENV_VARS,
        _workspace_default_roots(repository_ctx, ["cpp/build/_deps/json-src"]),
        [("include", "nlohmann/json.hpp")],
    )
    include_dir = include_dirs[0]
    _require_file_contains(
        repository_ctx,
        "nlohmann/json",
        include_dir,
        "nlohmann/detail/abi_macros.hpp",
        [
            "#define NLOHMANN_JSON_VERSION_MAJOR 3",
            "#define NLOHMANN_JSON_VERSION_MINOR 12",
            "#define NLOHMANN_JSON_VERSION_PATCH 0",
        ],
        _NLOHMANN_JSON_ENV_VARS,
    )
    build_file = _render_header_build_file(
        "json",
        "json_header_files",
        ["include/**"],
        ["include"],
    )
    _write_symlinked_repo(repository_ctx, [{"dest": "include", "source": include_dir}], build_file)


def _cutlass_repository_impl(repository_ctx):
    root, _ = _select_root_with_required_headers(
        repository_ctx,
        "CUTLASS",
        _CUTLASS_ENV_VARS,
        _workspace_default_roots(repository_ctx, ["tensorrt_llm/include/trtllm_gen_kernels/fmha/cutlass"]),
        [
            ("include", "cutlass/cutlass.h"),
            ("include", "cute/tensor.hpp"),
            ("tools/util/include", "cutlass/util/host_tensor.h"),
        ],
    )
    _require_file_contains(
        repository_ctx,
        "CUTLASS",
        root,
        "include/cutlass/version.h",
        [
            "#define CUTLASS_MAJOR 4",
            "#define CUTLASS_MINOR 4",
            "#define CUTLASS_PATCH 2",
        ],
        _CUTLASS_ENV_VARS,
    )
    build_file = _render_header_build_file(
        "cutlass",
        "cutlass_header_files",
        ["include/**", "tools/util/include/**"],
        ["include", "tools/util/include"],
        duplicate_names = ["cute"],
    )
    _write_symlinked_repo(
        repository_ctx,
        [
            {"dest": "include", "source": _join(root, "include")},
            {"dest": "tools", "source": _join(root, "tools")},
        ],
        build_file,
    )


_cuda_repository = repository_rule(
    implementation = _cuda_repository_impl,
    environ = _SYSTEM_DEP_ENV_VARS,
)

_tensorrt_repository = repository_rule(
    implementation = _tensorrt_repository_impl,
    environ = _SYSTEM_DEP_ENV_VARS,
)

_nccl_repository = repository_rule(
    implementation = _nccl_repository_impl,
    environ = _SYSTEM_DEP_ENV_VARS,
)

_mpi_repository = repository_rule(
    implementation = _mpi_repository_impl,
    environ = _SYSTEM_DEP_ENV_VARS,
)

_torch_repository = repository_rule(
    implementation = _torch_repository_impl,
    environ = _SYSTEM_DEP_ENV_VARS,
)

_workspace_repository_attrs = {
    "_workspace_file": attr.label(default = Label("//:MODULE.bazel")),
}

_nanobind_repository = repository_rule(
    implementation = _nanobind_repository_impl,
    attrs = _workspace_repository_attrs,
    environ = _SYSTEM_DEP_ENV_VARS,
)

_nlohmann_json_repository = repository_rule(
    implementation = _nlohmann_json_repository_impl,
    attrs = _workspace_repository_attrs,
    environ = _SYSTEM_DEP_ENV_VARS,
)

_cutlass_repository = repository_rule(
    implementation = _cutlass_repository_impl,
    attrs = _workspace_repository_attrs,
    environ = _SYSTEM_DEP_ENV_VARS,
)


def _system_deps_impl(module_ctx):
    _cuda_repository(name = "trtllm_local_cuda")
    _tensorrt_repository(name = "trtllm_local_tensorrt")
    _nccl_repository(name = "trtllm_local_nccl")
    _mpi_repository(name = "trtllm_local_mpi")
    _torch_repository(name = "trtllm_local_torch")
    _nanobind_repository(name = "trtllm_nanobind")
    _nlohmann_json_repository(name = "trtllm_nlohmann_json")
    _cutlass_repository(name = "trtllm_cutlass")


system_deps = module_extension(
    implementation = _system_deps_impl,
)
