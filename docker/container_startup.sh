#!/usr/bin/env bash
. /app/spack/share/spack/setup-env.sh
spack env activate /app
export LD_LIBRARY_PATH=/app/.spack-env/view/lib:$LD_LIBRARY_PATH
