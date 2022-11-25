from ubuntu:20.04 as builder

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y  software-properties-common patchelf build-essential ca-certificates coreutils curl environment-modules gfortran git gpg lsb-release file python3 python3-dev python3-distutils python3-venv unzip zip sudo && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-11 gfortran-11 g++-11 && \
    apt-get update -y && \
    apt-get clean -y && \
    groupadd demo && \
    useradd demo -d /home/demo -g demo && \
    mkdir /home/demo && \
    chown demo:demo /home/demo
RUN echo "demo    ALL=(ALL)       NOPASSWD: ALL" >> /etc/sudoers.d/demo
COPY --chown=demo:demo container_startup.sh /etc/profile.d/container_startup.sh
COPY --chown=demo:demo spack.yaml /app/
RUN su demo -c "git clone --depth=1 https://github.com/robertu94/spack_packages /app/robertu94_packages"
RUN su demo -c "git clone --depth=1 https://github.com/spack/spack /app/spack"
WORKDIR /app
USER demo
RUN . /etc/profile &&\
    spack compiler find && \
    spack repo add /app/robertu94_packages && \
    spack mirror add  E4S https://cache.e4s.io/22.11 && \
    spack buildcache keys --install --trust && \
    spack install --reuse --fail-fast
RUN find -L /app/.spack-env/view/* -type f -exec readlink -f '{}' \; | \
    grep -v 'nsight-compute' | \
    xargs file -i | \
    grep 'charset=binary' | \
    grep 'charset=binary' | \
    grep 'x-executable\|x-archive\|x-sharedlib' | \
    awk -F: '{print $1}' | xargs strip -s


from ubuntu:20.04 as final
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y  software-properties-common patchelf build-essential ca-certificates coreutils curl environment-modules gfortran git gpg lsb-release file python3 python3-dev python3-distutils python3-venv unzip zip sudo && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-11 gfortran-11 g++-11 && \
    apt-get clean -y && \
    groupadd demo && \
    useradd demo -d /home/demo -g demo && \
    mkdir /home/demo && \
    mkdir /zchecker && \
    chown demo:demo /zchecker /home/demo
RUN echo "demo    ALL=(ALL)       NOPASSWD: ALL" >> /etc/sudoers.d/demo
COPY container_startup.sh /etc/profile.d/container_startup.sh
COPY --from=builder --chown=demo:demo /app /app
USER demo
WORKDIR /app
CMD ["bash", "-l"]
