# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local system dependency repositories for TensorRT-LLM native Bazel builds."""

_CUDA_ENV_VARS = ["CUDA_HOME", "CUDA_PATH"]


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


def _find_header_dir(repository_ctx, root, header_candidates):
    for include_dir, header in header_candidates:
        header_path = _join(root, _header_path(include_dir, header))
        if repository_ctx.path(header_path).exists:
            return _join(root, include_dir)
    return None


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


def _lib_candidate_paths(root, lib_names, rel_lib_dirs, abs_lib_dirs):
    paths = []
    for rel_lib_dir in rel_lib_dirs:
        for lib_name in lib_names:
            paths.append(_join(root, rel_lib_dir + "/" + lib_name))
    for abs_lib_dir in abs_lib_dirs:
        for lib_name in lib_names:
            paths.append(_join(abs_lib_dir, lib_name))
    return paths


def _required_lib(repository_ctx, dep_name, target_name, root, lib_names, rel_lib_dirs, abs_lib_dirs, env_vars):
    for path in _lib_candidate_paths(root, lib_names, rel_lib_dirs, abs_lib_dirs):
        if repository_ctx.path(path).exists:
            return path
    fail(
        (
            "%s repository found root %s but could not find %s library (%s). " +
            "Set one of %s to the dependency root that contains this library."
        ) % (dep_name, root, target_name, ", ".join(lib_names), ", ".join(env_vars))
    )


def _format_list(items):
    if not items:
        return "[]"
    return "[\n" + "".join(["        \"%s\",\n" % item for item in items]) + "    ]"


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


def _render_cuda_build_file(libraries):
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
        "    includes = [\"include\"],",
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

    lines.extend([
        "cc_library(",
        "    name = \"cuda\",",
        "    deps = %s," % _format_list([":" + library["name"] for library in libraries]),
        ")",
        "",
    ])
    return "\n".join(lines)


def _write_cuda_repo(repository_ctx, include_links, libraries):
    libraries = _with_library_runfiles(repository_ctx, libraries)
    for include_link in include_links:
        repository_ctx.symlink(repository_ctx.path(include_link["source"]), include_link["dest"])
    repository_ctx.file("lib/.empty", "")
    for library in libraries:
        for link in library["links"]:
            repository_ctx.symlink(repository_ctx.path(link["source"]), link["dest"])
    repository_ctx.file("BUILD.bazel", _render_cuda_build_file(libraries))


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
        _lib_spec(
            "nvml",
            _required_lib(
                repository_ctx,
                "CUDA",
                "nvml",
                root,
                ["libnvidia-ml.so", "libnvidia-ml.so.1"],
                cuda_lib_dirs,
                cuda_driver_lib_dirs,
                _CUDA_ENV_VARS,
            ),
            "libnvidia-ml.so",
        ),
    ]
    _write_cuda_repo(repository_ctx, _include_dir_link(include_dir), libraries)


_cuda_repository = repository_rule(
    implementation = _cuda_repository_impl,
    environ = _CUDA_ENV_VARS,
)


def _system_deps_impl(module_ctx):
    _cuda_repository(name = "trtllm_local_cuda")


system_deps = module_extension(
    implementation = _system_deps_impl,
)
