import os
import tomllib
import typing as t
from dagger_clients.core import (
    Directory,
    Container,
    Workspace,
    core,
    CacheSharingMode,
    Platform,
)
from dagger import function, object_type, Doc, DefaultPath, check
from opentelemetry import trace

tracer = trace.get_tracer("dagger.io/util/parallel")

# SUPPORTED VERSIONS
UBUNTU = ["ubuntu:24.04", "ubuntu:26.04"]
FEDORA = ["fedora:44", "fedora:45"]
CENTOS = ["almalinux:9", "almalinux:10"]
PYTHON_VERSIONS = [
    "quay.io/pypa/manylinux_2_28_x86_64",
    "quay.io/pypa/manylinux_2_28_aarch64",
]


# DEFAULT VERSIONS
DEFAULT_BUILD_TYPE = "Release"
DEFAULT_UBUNTU = UBUNTU[-1]
DEFAULT_FEDORA = FEDORA[-1]
DEFAULT_CENTOS = CENTOS[-1]
DEFAULT_MANYLINUX = PYTHON_VERSIONS[-1]


@object_type
class Libpressio:
    """build and manage the LibPresiso CI/CD"""

    src: Directory

    @classmethod
    def create(cls, ws: Workspace):
        return cls(src=ws.directory("/"))

    async def manylinux_base_alma(
        self, image: str = "quay.io/pypa/manylinux_2_28_aarch64"
    ) -> Container:
        platform = Platform("linux/arm64" if "aarch64" in image else "linux/amd64")
        dnf_cache = core().cache_volume(f"manylinux-dnf-{platform}-{image}")
        ccache = core().cache_volume(f"manylinux-ccache-{platform}-{image}")
        return (
            core()
            .container(platform=platform)
            .from_(image)
            .with_mounted_cache(
                "/var/cache/dnf", dnf_cache, sharing=CacheSharingMode.LOCKED
            )
            .with_mounted_cache(
                "/var/ccache/cache", ccache, sharing=CacheSharingMode.LOCKED
            )
            .with_env_variable("CCACHE_DIR", "/var/ccache/cache")
            .with_exec(
                [
                    "dnf",
                    "install",
                    "-y",
                    "boost-devel",
                    "ccache",
                ]
            )
        )

    @function
    async def export_docs(self, build_jobs: int = -1) -> Directory:
        """export the documentation"""
        return (await self.build_docs(build_jobs)).directory("/build/html")

    @function
    async def build_docs(self, build_jobs: int = -1) -> Container:
        """build a container capable of building the documentation"""
        base = (await self.fedora_base(DEFAULT_FEDORA)).with_exec(
            ["dnf", "install", "-y", "doxygen", "graphviz"]
        )
        # build Debug because it builds faster
        return await self.common_build(
            base,
            tests=False,
            build_jobs=build_jobs,
            build_type="Debug",
            extra_cmake=["-DBUILD_DOCS=ON"],
            extra_build=["docs"],
        )

    @function
    async def build_manylinux(
        self,
        image: str = DEFAULT_MANYLINUX,
        tests: bool = False,
        build_jobs: int = -1,
        build_type: str = "Release",
        python_version: str = "python3.14",
    ) -> Container:
        """build all wheels for the specified python version"""
        has_ccache = True
        # Older images ship auditwheel 6.1, which drops $ORIGIN when the
        # library lives at the wheel root. Keep repair tooling independent
        # of the target Python (auditwheel no longer supports Python 3.8).
        container = (
            (await self.manylinux_base_alma(image))
            .with_exec(
                ["/opt/python/cp312-cp312/bin/python", "-m", "venv", "/auditwheel-venv"]
            )
            .with_exec(
                [
                    "/auditwheel-venv/bin/python",
                    "-m",
                    "pip",
                    "install",
                    "auditwheel>=6.4.2",
                ]
            )
        )
        stdcompat = (
            core().git("https://github.com/robertu94/std_compat").ref("master").tree()
        )
        google_test = (
            core().git("https://github.com/google/googletest").ref("v1.17.0").tree()
        )
        is_static = True
        generator = "Unix Makefiles"
        build_cmd = "make"
        build_jobs = len(os.sched_getaffinity(0)) if build_jobs == -1 else build_jobs
        r = (
            container.with_directory("/deps/stdcompat", stdcompat)
            .with_directory("/deps/googletest", google_test)
            .with_workdir("/deps/stdcompat")
            .with_exec(["mkdir", "/deps/googletest/build/", "/deps/stdcompat/build/"])
            .with_exec(
                [
                    "cmake",
                    "-S/deps/stdcompat",
                    "-B/deps/stdcompat/build/",
                    "-G",
                    generator,
                    *(
                        [
                            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                        ]
                        if has_ccache
                        else []
                    ),
                    # std_compat's tests fetch an old GoogleTest incompatible with CMake 4.
                    "-DBUILD_TESTING=OFF",
                    f"-DCMAKE_BUILD_TYPE={build_type}",
                    f"-DBUILD_SHARED_LIBS={is_static}",
                    f"-DSTDCOMPAT_CXX_VERSION=20",
                ]
            )
            .with_exec(
                [
                    "cmake",
                    "-S/deps/googletest",
                    "-B/deps/googletest/build/",
                    "-G",
                    generator,
                    *(
                        [
                            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                        ]
                        if has_ccache
                        else []
                    ),
                ]
            )
            .with_exec(
                [build_cmd, "-C", "/deps/googletest/build/", "-j", str(build_jobs)]
            )
            .with_exec(
                [build_cmd, "-C", "/deps/stdcompat/build/", "-j", str(build_jobs)]
            )
            .with_exec(["cmake", "--install", "/deps/stdcompat/build/"])
            .with_exec(["cmake", "--install", "/deps/googletest/build/"])
            .with_directory("/src/", self.src)
            .with_workdir("/src/")
            .with_exec(
                [
                    python_version,
                    "-m",
                    "build",
                    ".",
                    f"-Ccmake.build-type={build_type}",
                    "-Ccmake.define.CMAKE_CXX_COMPILER_LAUNCHER=ccache",
                    "-Ccmake.define.CMAKE_C_COMPILER_LAUNCHER=ccache",
                ]
            )
            .with_env_variable(
                "LD_LIBRARY_PATH", "/usr/local/lib64:${LD_LIBRARY_PATH}", expand=True
            )
            .with_exec(
                [
                    "sh",
                    "-c",
                    "/auditwheel-venv/bin/auditwheel repair dist/*.whl",
                ]
            )
        )
        if tests:
            # This build is separate from the wheel: test instrumentation is not shipped.
            r = (
                r.with_exec([python_version, "-m", "venv", "/test-venv"])
                .with_exec(
                    [
                        "/test-venv/bin/python",
                        "-m",
                        "pip",
                        "install",
                        "numpy",
                        "numcodecs",
                        "swig",
                    ]
                )
                .with_env_variable("PATH", "/test-venv/bin:${PATH}", expand=True)
                .with_exec(
                    [
                        "cmake",
                        "-S/src",
                        "-B/build",
                        "-DBUILD_TESTING=ON",
                        "-DBUILD_PYTHON_WRAPPER=ON",
                        "-DPython3_EXECUTABLE=/test-venv/bin/python",
                        f"-DCMAKE_BUILD_TYPE={build_type}",
                        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                    ]
                )
                .with_exec(
                    ["cmake", "--build", "/build", "--parallel", str(build_jobs)]
                )
                .with_workdir("/build")
            )
        return r

    async def test_manylinux_wheel(
        self, container: Container, image: str, python_version: str
    ) -> None:
        # A fresh runtime has neither /src nor the locally installed native dependencies.
        platform = await container.platform()
        runtime = (
            core()
            .container(platform=platform)
            .from_(image)
            .with_directory("/wheelhouse", container.directory("/src/wheelhouse"))
            .with_file(
                "/wheel-tests/test_python_wheel.py",
                self.src.file("test/test_python_wheel.py"),
            )
            .with_file(
                "/wheel-tests/test_python_to_cpp.py",
                self.src.file("test/test_python_to_cpp.py"),
            )
            .with_workdir("/wheel-tests")
            .with_exec([python_version, "-m", "venv", "/wheel-venv"])
            .with_exec(
                ["sh", "-c", "/wheel-venv/bin/python -m pip install /wheelhouse/*.whl"]
            )
            .with_exec(["/wheel-venv/bin/python", "-m", "pip", "check"])
            .with_exec(["/wheel-venv/bin/python", "-I", "test_python_wheel.py"])
            .with_exec(["/wheel-venv/bin/python", "-I", "test_python_to_cpp.py"])
        )
        await runtime.sync()

    async def ubuntu_base(self, image: str = "ubuntu:24.04") -> Container:
        apt_cache = core().cache_volume(f"apt-{image}")
        ccache = core().cache_volume(f"ccache-{image}")
        return (
            core()
            .container()
            .from_(image)
            .with_exec(["rm", "/etc/apt/apt.conf.d/docker-clean"])
            .with_mounted_cache("/var/apt/cache", apt_cache)
            .with_mounted_cache("/var/ccache/cache", ccache)
            .with_env_variable("CCACHE_DIR", "/var/ccache/cache")
            .with_exec(["apt-get", "update", "-y"])
            .with_env_variable("DEBIAN_FRONTEND", "noninteractive")
            .with_exec(
                [
                    "apt-get",
                    "install",
                    "-y",
                    "cmake",
                    "gcc",
                    "g++",
                    "ninja-build",
                    "ccache",
                    "git",
                    "pkg-config",
                ]
            )
            .without_env_variable("DEBIAN_FRONTEND")
        )

    async def centos_base(
        self, image: str = "almalinux:9", powertools_repo: str = "powertools"
    ) -> Container:
        """Build a CentOS style container base image"""
        dnf_cache = core().cache_volume(f"centos-dnf-{image}")
        ccache = core().cache_volume(f"centos-ccache-{image}")
        return (
            core()
            .container()
            .from_(image)
            .with_mounted_cache(
                "/var/cache/dnf", dnf_cache, sharing=CacheSharingMode.LOCKED
            )
            .with_mounted_cache(
                "/var/ccache/cache", ccache, sharing=CacheSharingMode.LOCKED
            )
            .with_env_variable("CCACHE_DIR", "/var/ccache/cache")
            .with_exec(
                [
                    "dnf",
                    "install",
                    "-y",
                    "dnf-plugins-core",
                ]
            )
            .with_exec(
                [
                    "dnf",
                    "install",
                    "-y",
                    "epel-release",
                ]
            )
            .with_exec(
                [
                    "dnf",
                    "config-manager",
                    "--set-enabled",
                    powertools_repo,
                ]
            )
            .with_exec(
                [
                    "dnf",
                    "install",
                    "-y",
                    "cmake",
                    "gcc",
                    "gcc-c++",
                    "ninja-build",
                    "boost-devel",
                    "ccache",
                    "git",
                    "pkg-config",
                ]
            )
        )

    async def fedora_base(self, image: str = "fedora:42"):
        """Build a Fedora style container base image"""
        dnf_cache = core().cache_volume(f"fedora-dnf-{image}")
        ccache = core().cache_volume(f"fedora-ccache-{image}")
        return (
            core()
            .container()
            .from_(image)
            .with_mounted_cache(
                "/var/cache/dnf", dnf_cache, sharing=CacheSharingMode.LOCKED
            )
            .with_mounted_cache(
                "/var/ccache/cache", ccache, sharing=CacheSharingMode.LOCKED
            )
            .with_env_variable("CCACHE_DIR", "/var/ccache/cache")
            .with_exec(
                [
                    "dnf",
                    "install",
                    "-y",
                    "cmake",
                    "gcc",
                    "g++",
                    "ninja-build",
                    "ccache",
                    "git",
                    "pkg-config",
                ]
            )
        )

    async def common_test(
        self, container: Container, name: str, filter_rgx: str | None = None
    ) -> Container:
        args = ["ctest", "--output-on-failure", "--output-junit", "/junit"]
        if filter_rgx is not None:
            args.extend(["-R", filter_rgx])
        return await container.with_exec(args)

    async def common_build(
        self,
        container: Container,
        static_build: bool = False,
        generator: str = "Ninja",
        has_ccache: bool = True,
        tests: bool = False,
        build_jobs: int = -1,
        build_type: str = "Release",
        extra_cmake: list[str] | None = None,
        extra_build: list[str] | None = None,
        build_testing=False,
    ) -> Container:
        extra_cmake = extra_cmake or []
        extra_build = extra_build or []
        build_cmd = {"Ninja": "ninja", "Unix Makefiles": "make"}[generator]
        is_static = "OFF" if static_build else "ON"
        build_tests = "ON" if tests else "OFF"
        build_jobs = len(os.sched_getaffinity(0)) if build_jobs == -1 else build_jobs
        stdcompat = (
            core().git("https://github.com/robertu94/std_compat").ref("master").tree()
        )
        google_test = (
            core().git("https://github.com/google/googletest").ref("v1.17.0").tree()
        )
        r = (
            container.with_directory("/deps/stdcompat", stdcompat)
            .with_directory("/deps/googletest", google_test)
            .with_workdir("/deps/stdcompat")
            .with_exec(["mkdir", "/deps/stdcompat/build/", "/deps/googletest/build/"])
            .with_exec(
                [
                    "cmake",
                    "-S/deps/stdcompat",
                    "-B/deps/stdcompat/build/",
                    "-G",
                    generator,
                    *(
                        [
                            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                        ]
                        if has_ccache
                        else []
                    ),
                    f"-DBUILD_TESTING={build_testing}",
                    f"-DCMAKE_BUILD_TYPE={build_type}",
                    f"-DBUILD_SHARED_LIBS={is_static}",
                ]
            )
            .with_exec(
                [build_cmd, "-C", "/deps/stdcompat/build/", "-j", str(build_jobs)]
            )
            .with_exec(["cmake", "--install", "/deps/stdcompat/build/"])
            .with_exec(
                [
                    "cmake",
                    "-S/deps/googletest",
                    "-B/deps/googletest/build/",
                    "-G",
                    generator,
                    *(
                        [
                            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                        ]
                        if has_ccache
                        else []
                    ),
                ]
            )
            .with_exec(
                [build_cmd, "-C", "/deps/googletest/build/", "-j", str(build_jobs)]
            )
            .with_exec(["cmake", "--install", "/deps/googletest/build/"])
            .with_directory("/src/", self.src)
            .with_workdir("/src/")
            .with_exec(["mkdir", "/build"])
            .with_exec(
                [
                    "cmake",
                    "-S/src",
                    "-B/build/",
                    "-G",
                    generator,
                    *(
                        [
                            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                            "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                        ]
                        if has_ccache
                        else []
                    ),
                    f"-DBUILD_TESTING={build_tests}",
                    f"-DCMAKE_BUILD_TYPE={build_type}",
                    f"-DBUILD_SHARED_LIBS={is_static}",
                    *extra_cmake,
                ]
            )
            .with_exec(
                [build_cmd, "-C", "/build/", "-j", str(build_jobs), *extra_build]
            )
            .with_workdir("/build/")
        )
        return r

    @function
    async def build_centos(
        self,
        version: str = DEFAULT_CENTOS,
        tests: bool = False,
        build_jobs: int = -1,
        build_type: str = DEFAULT_BUILD_TYPE,
    ) -> Container:
        "build a CentOS container with libpressio"
        if int(version.split(":")[1]) <= 8:
            return await self.common_build(
                await self.centos_base(version),
                tests=tests,
                build_jobs=build_jobs,
                build_type=build_type,
            )
        else:
            return await self.common_build(
                await self.centos_base(version, powertools_repo="crb"),
                tests=tests,
                build_jobs=build_jobs,
                build_type=build_type,
            )

    @function
    async def build_ubuntu(
        self,
        version: str = DEFAULT_UBUNTU,
        tests: bool = False,
        build_jobs: int = -1,
        build_type: str = DEFAULT_BUILD_TYPE,
    ) -> Container:
        "build a Ubuntu container with libpressio"
        return await self.common_build(
            await self.ubuntu_base(version),
            tests=tests,
            build_jobs=build_jobs,
            build_type=build_type,
        )

    @function
    async def build_fedora(
        self,
        version: str = DEFAULT_FEDORA,
        tests: bool = False,
        build_jobs: int = -1,
        build_type: str = DEFAULT_BUILD_TYPE,
    ) -> Container:
        "build a Fedora container with libpressio"
        return await self.common_build(
            await self.fedora_base(version),
            tests=tests,
            build_jobs=build_jobs,
            build_type=build_type,
        )

    @function
    @check
    async def test_manylinux(
        self,
        version: str = "",
        build_jobs: int = -1,
        filter_rgx: str | None = None,
        build_type: str = DEFAULT_BUILD_TYPE,
        python_version: str = "",
    ) -> None:
        "Test repaired wheels and native bindings on the current architecture using the newest supported Python by default."
        metadata = tomllib.loads(await self.src.file("pyproject.toml").contents())
        prefix = "Programming Language :: Python :: 3."
        versions = [
            "3." + c.removeprefix(prefix)
            for c in metadata["project"]["classifiers"]
            if c.startswith(prefix)
        ]
        if python_version:
            if python_version not in versions:
                raise ValueError(
                    f"Python {python_version} is not advertised in pyproject.toml"
                )
            versions = [python_version]
        if version:
            images = [version]
        else:
            platform = await core().default_platform()
            architecture = str(platform).split("/")[1]
            manylinux_architecture = {"amd64": "x86_64", "arm64": "aarch64"}.get(
                architecture
            )
            if manylinux_architecture is None:
                raise ValueError(f"Unsupported manylinux platform: {platform}")
            images = [
                image
                for image in PYTHON_VERSIONS
                if image.endswith(f"_{manylinux_architecture}")
            ]
        for image in images:
            image_versions = versions
            if not python_version:
                platform = Platform(
                    "linux/arm64" if "aarch64" in image else "linux/amd64"
                )
                interpreters = await (
                    core()
                    .container(platform=platform)
                    .from_(image)
                    .directory("/opt/python")
                    .entries()
                )
                supported_versions = [
                    py
                    for py in versions
                    if f"cp{py.replace('.', '')}-cp{py.replace('.', '')}"
                    in interpreters
                ]
                if not supported_versions:
                    raise ValueError(
                        f"No advertised Python versions are available in {image}"
                    )
                image_versions = [
                    max(
                        supported_versions,
                        key=lambda py: tuple(map(int, py.split("."))),
                    )
                ]
            for py in image_versions:
                # Current PyPA images no longer ship CPython 3.8.
                build_image = (
                    image + ":2024.11.30-1"
                    if py == "3.8" and ":" not in image.rsplit("/", 1)[-1]
                    else image
                )
                abi = "cp" + py.replace(".", "")
                interpreter = f"/opt/python/{abi}-{abi}/bin/python"
                with tracer.start_as_current_span(
                    f"manylinux {build_image} Python {py}"
                ):
                    container = await self.build_manylinux(
                        build_image,
                        tests=True,
                        build_jobs=build_jobs,
                        build_type=build_type,
                        python_version=interpreter,
                    )
                    args = [
                        "ctest",
                        "--test-dir",
                        "/build",
                        "--output-on-failure",
                        "--no-tests=error",
                        "--output-junit",
                        "/junit",
                    ]
                    if filter_rgx is not None:
                        args.extend(["-R", filter_rgx])
                    await container.with_exec(args).sync()
                    await self.test_manylinux_wheel(container, build_image, interpreter)

    @function
    @check
    async def test_centos(
        self,
        version: str = DEFAULT_CENTOS,
        build_jobs: int = -1,
        filter_rgx: str | None = None,
        build_type: str = DEFAULT_BUILD_TYPE,
    ) -> None:
        "run tests on a CentOS container"
        (
            await self.common_test(
                await self.build_centos(
                    version, tests=True, build_jobs=build_jobs, build_type=build_type
                ),
                version,
                filter_rgx=filter_rgx,
            )
        ).sync()

    @function
    @check
    async def test_fedora(
        self,
        version: str = DEFAULT_FEDORA,
        build_jobs: int = -1,
        filter_rgx: str | None = None,
        build_type: str = DEFAULT_BUILD_TYPE,
    ) -> None:
        "run tests on a Fedora container"
        (
            await self.common_test(
                await self.build_fedora(
                    version, tests=True, build_jobs=build_jobs, build_type=build_type
                ),
                version,
                filter_rgx=filter_rgx,
            )
        ).sync()

    @function
    @check
    async def test_ubuntu(
        self,
        version: str = DEFAULT_UBUNTU,
        build_jobs: int = -1,
        filter_rgx: str | None = None,
        build_type: str = DEFAULT_BUILD_TYPE,
    ) -> None:
        "run tests on a Ubuntu container"
        (
            await self.common_test(
                await self.build_ubuntu(
                    version, tests=True, build_jobs=build_jobs, build_type=build_type
                ),
                version,
                filter_rgx=filter_rgx,
            )
        ).sync()
