all:
	/usr/local/cuda-10.1/bin/nvcc -lcuda -lcublas main.cu layers/*.cu -o main.out  -arch=compute_35 -Wno-deprecated-gpu-targets
debug:
	/usr/local/cuda-10.1/bin/nvcc -lcuda -lcublas -g -G main.cu layers/*.cu -o main.out  -arch=compute_35 -Wno-deprecated-gpu-targets
	cuda-gdb ./main.out
run:
	./main.out
clean:
	rm main.out
