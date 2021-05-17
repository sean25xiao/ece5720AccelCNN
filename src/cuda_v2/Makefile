all:
	/usr/local/cuda-10.1/bin/nvcc -lcuda -lcublas -g -G test.cu layers/*.cu -o test.out  -arch=compute_35 -Wno-deprecated-gpu-targets
debug:
	cuda-gdb ./test.out
run:
	./test.out
clean:
	rm test.out
