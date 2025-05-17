#!/bin/bash
BUILD_DIR="build/stream"
BUFFER_SIZES="8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912 1073741824"

for exe in "stream-base" "stream-omp-host" "stream-cuda"; do
	echo "Testing: $exe"
	for nx in $BUFFER_SIZES; do
		echo "NX=$nx:"
		"$BUILD_DIR/$exe" $nx
	done
	echo ""
done
