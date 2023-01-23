#!/usr/bin/env python

import sys
import argparse
import dagger
import os
import anyio


def many_linux(client):
    pass

    return (
        client.container()
        .from_("quay.io/pypa/manylinux2014_x86_64")
    )

def fedora_base(client, image):
    dnf_cache = client.cache_volume(f"dnf-{image}")
    ccache = client.cache_volume(f"ccache-{image}")
    return (
        client.container()
        .from_(image)
        .with_mounted_cache("/var/cache/dnf", dnf_cache)
        .with_mounted_cache("/var/ccache/cache", ccache)
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

def centos_base(client, image):
    dnf_cache = client.cache_volume(f"dnf-{image}")
    ccache = client.cache_volume(f"ccache-{image}")
    container = (
        client.container()
        .from_(image)
        .with_mounted_cache("/var/cache/dnf", dnf_cache)
        .with_mounted_cache("/var/ccache/cache", ccache)
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
                "powertools",
            ]
        )
    )
    container = (
        container.with_exec(
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
    return container

def ubuntu_base(client, image):
    apt_cache = client.cache_volume(f"apt-{image}")
    ccache = client.cache_volume(f"ccache-{image}")
    return (
        client.container()
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


async def test(args):
    async with dagger.Connection(dagger.Config(log_output=sys.stderr)) as client:
        # get reference to the local project
        src = client.host().directory(
            ".", exclude=["build*", "test/CMakeFiles*", "CMakeFiles*", "Testing*"]
        )
        stdcompat = (
            client.git("https://github.com/robertu94/std_compat")
            .branch("master")
            .tree()
        )

        def common_steps(container):
            return (
                container.with_directory("/deps/stdcompat", stdcompat)
                .with_workdir("/deps/stdcompat")
                .with_exec(["mkdir", "/deps/stdcompat/build/"])
                .with_exec(
                    [
                        "cmake",
                        "-S/deps/stdcompat",
                        "-B/deps/stdcompat/build/",
                        "-G",
                        "Ninja",
                        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                        "-DBUILD_TESTING=OFF",
                    ]
                )
                .with_exec(["cmake", "--build", "/deps/stdcompat/build/"])
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
                        "Ninja",
                        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                        "-DBUILD_TESTING=ON",
                    ]
                )
                .with_exec(
                    [
                        "cmake",
                        "--build",
                        "/build/",
                        "-j",
                        str(len(os.sched_getaffinity(0))),
                    ]
                )
                .with_workdir("/build/")
                .with_exec(["ctest", "--output-on-failure"])
                .with_exec(["cmake", "--install", "/build/"])
            )

        async def test_ubuntu(version):
            return await common_steps(ubuntu_base(client, version)).exit_code()

        async def test_fedora(version):
            return await common_steps(fedora_base(client, version)).exit_code()

        async def test_centos(version):
            return await common_steps(centos_base(client, version)).exit_code()

        async with anyio.create_task_group() as tg:
            tg.start_soon(many_linux, client)
            pass
            # for version in ["almalinux:8"]:
            #     tg.start_soon(test_centos, version)
            # for version in ["fedora:36", "fedora:37"]:
            #     tg.start_soon(test_fedora, version)
            # for version in ["ubuntu:20.04", "ubuntu:22.04"]:
            #     tg.start_soon(test_ubuntu, version)

        print("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    anyio.run(test, args)
