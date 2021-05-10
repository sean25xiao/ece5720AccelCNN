test:
	/usr/local/cuda-10.1/bin/nvcc -lcuda -lcublas main.cu model.cu layers/*.cu -o main.out  -arch=compute_35 -Wno-deprecated-gpu-targets
run:
	./test.out
clean:
	rm test.out
