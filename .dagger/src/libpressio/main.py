import asyncio
import os
import typing as t
import pandas as pd
import dagger
import xml.etree.ElementTree as ET
from rich.console  import Console
from rich.table  import Table
from collections import abc
from dagger import dag, function, object_type, Doc, DefaultPath, ReturnType
from contextlib import asynccontextmanager
from opentelemetry import trace
tracer = trace.get_tracer("dagger.io/util/parallel")

# SUPPORTED VERSIONS
UBUNTU = ["ubuntu:22.04", "ubuntu:24.04"]
FEDORA = ["fedora:41", "fedora:42"]
CENTOS = ["almalinux:8", "almalinux:9"]
PYTHON_VERSIONS = ["python3.14", "python3.13", "python3.12", "python3.11", "python3.10", "python3.9"]

#DEFAULT VERSIONS
DEFAULT_BUILD_TYPE = "Release"
DEFAULT_UBUNTU = UBUNTU[-1]
DEFAULT_FEDORA = FEDORA[-1]
DEFAULT_CENTOS = CENTOS[-1]

class ParallelTaskContext:
    def __init__(self, max_width: int|None = None):
        self.pending: set[asyncio.Task[t.Any]] = set()
        self.running : set[asyncio.Task[t.Any]]= set()
        self.completed : list[asyncio.Task[t.Any]]= []
        self.max_width: int|None = max_width

    def create_job[T](self, name :str, coro: abc.Coroutine[None, None, T]) -> asyncio.Task[T]:
        with tracer.start_as_current_span(name):
            task = asyncio.Task(coro)
            self.pending.add(task)
            return task
    async def run(self):
        if self.max_width is None:
            self.running = self.pending.copy()
            self.pending.clear()
        else:
            for _ in range(self.max_width):
                try:
                    self.running.add(self.pending.pop())
                except IndexError:
                    break
        while len(self.pending) > 0:
            done, self.running = await asyncio.wait(self.running, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                try:
                    await d.result()
                except:
                    pass
            if len(self.pending) > 0:
                for _ in range(t.cast(int,self.max_width) - len(self.running)):
                    try:
                        self.running.add(self.pending.pop())
                    except IndexError:
                        break
        while len(self.running) > 0:
            done, self.running = await asyncio.wait(self.running, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                try:
                    await d.result()
                except:
                    pass

@asynccontextmanager
async def ParallelTasks(max_width: int|None = None):
    tc =  ParallelTaskContext(max_width)
    yield tc
    await tc.run()


def format_table(df: pd.DataFrame):
    df = df[["version", "classname", "output", "passed"]]
    failures = df[df.passed != True]
    if len(failures) > 0:
        table = Table(title="test failures")
        table.add_column("index")
        for col in failures.columns:
            table.add_column(col)
        row : tuple[str]
        for row in failures.itertuples(name=None):
            table.add_row(*[str(r) for r in row])
        console = Console(width=120)
        with console.capture() as capture:
            console.print(table)
        return capture.get()
    else:
        return f"All {len(df)} Tests Passed!"


@object_type
class Libpressio:

    async def manylinux_base_alma(self, image: str = "quay.io/pypa/manylinux_2_28_x86_64") -> dagger.Container:
        dnf_cache = dag.cache_volume(f"manylinux-dnf-{image}")
        ccache = dag.cache_volume(f"manylinux-ccache-{image}")
        return (
            dag.container()
            .from_(image)
            .with_mounted_cache("/var/cache/dnf", dnf_cache, sharing=dagger.CacheSharingMode.LOCKED)
            .with_mounted_cache("/var/ccache/cache", ccache, sharing=dagger.CacheSharingMode.LOCKED)
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
    async def export_docs(self, src: t.Annotated[dagger.Directory, Doc("Source Directory"), DefaultPath("/")], build_jobs: int = -1) -> dagger.Directory:
        """export the documentation"""
        return (await self.build_docs(src, build_jobs)).directory("/build/html")

    @function
    async def build_docs(self, src: t.Annotated[dagger.Directory, Doc("Source Directory"), DefaultPath("/")], build_jobs: int = -1) -> dagger.Container:
        """build a container capable of building the documentation"""
        base =  ((await self.fedora_base(DEFAULT_FEDORA))
                 .with_exec( [ "dnf", "install", "-y", "doxygen", "graphviz" ])
                 )
        return await self.common_build(base , src=src, tests=False, build_jobs=build_jobs, build_type="Debug",
                                       extra_cmake=["-DBUILD_DOCS=ON"], extra_build=["docs"])

    @function
    async def manylinux_all(self, src: t.Annotated[dagger.Directory, Doc("Source Directory"), DefaultPath("/")], image: str = "quay.io/pypa/manylinux_2_28_x86_64", build_jobs: int = -1, build_type:str="Release"):
        """build all wheels for all supported python versions"""
        try:
            n_jobs = len(os.sched_getaffinity(0)) 
            n_tasks = len(PYTHON_VERSIONS)
            async with ParallelTasks() as tg:
                for py_version in PYTHON_VERSIONS:
                    tg.create_job(py_version, self.manylinux(src, image=image, build_jobs=n_jobs//n_tasks, build_type=build_type, python_version=py_version))
        except* asyncio.CancelledError:
            pass



    @function
    async def manylinux(self, src: t.Annotated[dagger.Directory, Doc("Source Directory"), DefaultPath("/")], image: str = "quay.io/pypa/manylinux_2_28_x86_64", build_jobs: int = -1, build_type:str="Release", python_version: str = "python3.14") -> dagger.Container:
        """build all wheels for the specified python version"""
        has_ccache=True
        container = await self.manylinux_base_alma(image)
        stdcompat = dag.git("https://github.com/robertu94/std_compat").ref("master").tree()
        is_static = True
        generator = "Unix Makefiles"
        build_cmd = "make"
        build_jobs = len(os.sched_getaffinity(0)) if build_jobs == -1 else build_jobs
        r = (
            container.with_directory("/deps/stdcompat", stdcompat)
            .with_workdir("/deps/stdcompat")
            .with_exec(["mkdir", "/deps/stdcompat/build/"])
            .with_exec(
                [
                    "cmake",
                    "-S/deps/stdcompat",
                    "-B/deps/stdcompat/build/",
                    "-G",
                    generator,
                    *(["-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                       "-DCMAKE_C_COMPILER_LAUNCHER=ccache"] if has_ccache else []),
                    "-DBUILD_TESTING=OFF",
                    f"-DCMAKE_BUILD_TYPE={build_type}",
                    f"-DBUILD_SHARED_LIBS={is_static}",
                    f"-DSTDCOMPAT_CXX_VERSION=20",
                ]
            )
            .with_exec([build_cmd, "-C", "/deps/stdcompat/build/", "-j",  str(build_jobs)])
            .with_exec(["cmake", "--install", "/deps/stdcompat/build/"])
            .with_directory("/src/", src)
            .with_workdir("/src/")
            .with_exec([python_version, "-m", "build", "."])
            .with_env_variable("LD_LIBRARY_PATH", "/usr/local/lib64:${LD_LIBRARY_PATH}", expand=True)
            .with_exec(["sh", "-c", "auditwheel repair dist/*.whl",])
            )
        return r
        
    async def ubuntu_base(self, image: str = "ubuntu:24.04") -> dagger.Container:
        apt_cache = dag.cache_volume(f"apt-{image}")
        ccache = dag.cache_volume(f"ccache-{image}")
        return (
            dag.container()
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


    async def centos_base(self, image: str = "almalinux:9", powertools_repo: str="powertools") ->  dagger.Container:
        """Build a CentOS style container base image"""
        dnf_cache = dag.cache_volume(f"centos-dnf-{image}")
        ccache = dag.cache_volume(f"centos-ccache-{image}")
        return (
            dag.container()
            .from_(image)
            .with_mounted_cache("/var/cache/dnf", dnf_cache, sharing=dagger.CacheSharingMode.LOCKED)
            .with_mounted_cache("/var/ccache/cache", ccache, sharing=dagger.CacheSharingMode.LOCKED)
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
        dnf_cache = dag.cache_volume(f"fedora-dnf-{image}")
        ccache = dag.cache_volume(f"fedora-ccache-{image}")
        return (
            dag.container()
            .from_(image)
            .with_mounted_cache("/var/cache/dnf", dnf_cache, sharing=dagger.CacheSharingMode.LOCKED)
            .with_mounted_cache("/var/ccache/cache", ccache, sharing=dagger.CacheSharingMode.LOCKED)
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
    async def common_test(self, container: dagger.Container, name: str, filter_rgx: str|None =None) -> pd.DataFrame:
        args = ["ctest", "--output-on-failure", "--output-junit", "/junit"]
        if filter_rgx is not None:
            args.extend(["-R",filter_rgx])
        r = await container.with_exec(args, expect=ReturnType.ANY)
        junit_xml = await r.file("/junit").contents()
        tree = ET.ElementTree(ET.fromstring(junit_xml))
        root = tree.getroot()
        if root is None:
            return pd.DataFrame()
        rows: list[dict[str, t.Any]] = []
        for testcase in root.iter("testcase"):
            skipped = False
            skip_elm = testcase.find("skipped")
            if skip_elm is not None:
                skipped = skip_elm.attrib.get("message", "unknown")
            passed = True
            failed = False
            fail_elm = testcase.find("failure")
            if fail_elm is not None:
                passed = False
                failed = fail_elm.attrib.get("message", "unknown")
            rows.append({
                "version" : name,
                "casename" : testcase.attrib.get("name"),
                "classname" : testcase.attrib.get("classname"),
                "time" : float(testcase.attrib.get("time", 0.0)),
                "output" : testcase.findtext("system-out", ""),
                "passed" : passed,
                "failed" : failed,
                "skipped" : skipped,
            })

        df = pd.DataFrame(rows)
        return df
        

    async def common_build(self, container: dagger.Container, src: t.Annotated[dagger.Directory, Doc("source directory")], static_build: bool = False, generator: str = "Ninja",
                           has_ccache: bool = True, tests: bool = False, build_jobs:int= -1, build_type:str="Release", extra_cmake: list[str]|None = None,
                           extra_build: list[str]|None = None
                           ) -> dagger.Container:
        extra_cmake = extra_cmake or []
        extra_build = extra_build or []
        build_cmd = {
            "Ninja": "ninja",
            "Unix Makefiles": "make"
        }[generator]
        is_static = "OFF" if static_build else "ON"
        build_tests = "ON" if tests else "OFF"
        build_jobs = len(os.sched_getaffinity(0)) if build_jobs == -1 else build_jobs
        stdcompat = dag.git("https://github.com/robertu94/std_compat").ref("master").tree()
        r = (
            container.with_directory("/deps/stdcompat", stdcompat)
            .with_workdir("/deps/stdcompat")
            .with_exec(["mkdir", "/deps/stdcompat/build/"])
            .with_exec(
                [
                    "cmake",
                    "-S/deps/stdcompat",
                    "-B/deps/stdcompat/build/",
                    "-G",
                    generator,
                    *(["-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                       "-DCMAKE_C_COMPILER_LAUNCHER=ccache"] if has_ccache else []),
                    "-DBUILD_TESTING=OFF",
                    f"-DCMAKE_BUILD_TYPE={build_type}",
                    f"-DBUILD_SHARED_LIBS={is_static}",
                ]
            )
            .with_exec([build_cmd, "-C", "/deps/stdcompat/build/", "-j",  str(build_jobs)])
            .with_exec(["cmake", "--install", "/deps/stdcompat/build/"])
            .with_directory("/src/", src)
            .with_workdir("/src/")
            .with_exec(["mkdir", "/build"])
            .with_exec(
                [
                    "cmake",
                    "-S/src",
                    "-B/build/",
                    "-G",
                    generator,
                    *(["-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                       "-DCMAKE_C_COMPILER_LAUNCHER=ccache"] if has_ccache else []),
                    f"-DBUILD_TESTING={build_tests}",
                    f"-DCMAKE_BUILD_TYPE={build_type}",
                    f"-DBUILD_SHARED_LIBS={is_static}",
                    *extra_cmake
                ]
            )
            .with_exec(
                [
                    build_cmd,
                    "-C",
                    "/build/",
                    "-j",
                    str(build_jobs),
                    *extra_build
                ]
            )
            .with_workdir("/build/")
        )
        return r

    @function
    async def build_centos(self, src: t.Annotated[dagger.Directory, DefaultPath("/")], version:str= DEFAULT_CENTOS, tests: bool = False, build_jobs:int = -1, build_type:str=DEFAULT_BUILD_TYPE) -> dagger.Container:
        "build a CentOS container with libpressio"
        if int(version.split(":")[1]) <= 8:
            return await self.common_build(await self.centos_base(version), src=src, tests=tests, build_jobs=build_jobs, build_type=build_type)
        else:
            return await self.common_build(await self.centos_base(version, powertools_repo="crb"), src=src, tests=tests, build_jobs=build_jobs, build_type=build_type)

    @function
    async def build_ubuntu(self, src: t.Annotated[dagger.Directory, DefaultPath("/")], version: str=DEFAULT_UBUNTU, tests:bool=False, build_jobs:int = -1, build_type:str=DEFAULT_BUILD_TYPE) -> dagger.Container:
        "build a Ubuntu container with libpressio"
        return await self.common_build(await self.ubuntu_base(version), src=src, tests=tests, build_jobs=build_jobs, build_type=build_type)

    @function
    async def build_fedora(self, src: t.Annotated[dagger.Directory, DefaultPath("/")],version: str = DEFAULT_FEDORA, tests:bool = False, build_jobs: int = -1, build_type:str=DEFAULT_BUILD_TYPE) -> dagger.Container:
        "build a Fedora container with libpressio"
        return await self.common_build(await self.fedora_base(version), src=src, tests=tests, build_jobs=build_jobs, build_type=build_type)


    async def test_centos_df(self, src: t.Annotated[dagger.Directory, DefaultPath("/")], version: str=DEFAULT_CENTOS, build_jobs: int = -1, filter_rgx: str|None = None, build_type:str=DEFAULT_BUILD_TYPE) -> pd.DataFrame:
        "run tests on a CentOS container"
        return await self.common_test(await self.build_centos(src, version, tests=True, build_jobs=build_jobs, build_type=build_type), version, filter_rgx=filter_rgx)


    @function
    async def test_centos(self, src: t.Annotated[dagger.Directory, DefaultPath("/")], version: str=DEFAULT_CENTOS, build_jobs: int = -1, filter_rgx: str|None = None, build_type:str=DEFAULT_BUILD_TYPE) -> str:
        "run tests on a CentOS container"
        return format_table(await self.test_centos_df(src, version, build_jobs, filter_rgx=filter_rgx, build_type=build_type))

    async def test_fedora_df(self, src: t.Annotated[dagger.Directory, DefaultPath("/")], version: str = DEFAULT_FEDORA, build_jobs: int = -1, filter_rgx: str|None = None, build_type:str=DEFAULT_BUILD_TYPE) -> pd.DataFrame:
        "run tests on a Fedora container"
        return await self.common_test(await self.build_fedora(src, version, tests=True, build_jobs=build_jobs, build_type=build_type), version, filter_rgx=filter_rgx)

    @function
    async def test_fedora(self, src: t.Annotated[dagger.Directory, DefaultPath("/")], version: str = DEFAULT_FEDORA, build_jobs: int = -1, filter_rgx: str|None = None, build_type:str=DEFAULT_BUILD_TYPE) -> str:
        "run tests on a Fedora container"
        return format_table(await self.test_fedora_df(src, version, build_jobs, filter_rgx=filter_rgx, build_type=build_type))

    async def test_ubuntu_df(self, src: t.Annotated[dagger.Directory, DefaultPath("/")], version: str= DEFAULT_UBUNTU, build_jobs: int = -1, filter_rgx: str|None = None, build_type:str=DEFAULT_BUILD_TYPE) -> pd.DataFrame:
        "run tests on a Ubuntu container"
        return await self.common_test(await self.build_ubuntu(src, version, tests=True, build_jobs=build_jobs, build_type=build_type), version, filter_rgx=filter_rgx)
    
    @function
    async def test_ubuntu(self, src: t.Annotated[dagger.Directory, DefaultPath("/")], version: str= DEFAULT_UBUNTU, build_jobs: int = -1, filter_rgx: str|None = None, build_type:str=DEFAULT_BUILD_TYPE) -> str:
        "run tests on a Ubuntu container"
        return format_table(await self.test_ubuntu_df(src, version, build_jobs, filter_rgx=filter_rgx, build_type=build_type))

    @function
    async def test_all(self, src: t.Annotated[dagger.Directory, DefaultPath("/")]) -> str:
        """run all tests"""
        results: list[asyncio.Task[pd.DataFrame]] = []
        try:
            n_jobs = len(os.sched_getaffinity(0)) 
            n_tasks = len(CENTOS) + len(FEDORA) + len(UBUNTU)
            async with ParallelTasks() as tg:
                for version in CENTOS:
                    results.append(tg.create_job(f"test {version}", self.test_centos_df(src,version, build_jobs=n_jobs//n_tasks)))
                for version in FEDORA:
                    results.append(tg.create_job(f"test {version}", self.test_fedora_df(src,version, build_jobs=n_jobs//n_tasks)))
                for version in UBUNTU:
                    results.append(tg.create_job(f"test {version}", self.test_ubuntu_df(src,version, build_jobs=n_jobs//n_tasks)))
        except* asyncio.CancelledError:
            pass
        return format_table(pd.concat(r.result() for r in results))


    @function
    async def play_fifo(self, svc: dagger.Service) -> dagger.Container:
        return await dag.container().from_("fedora:42").with_service_binding("svc", svc)

