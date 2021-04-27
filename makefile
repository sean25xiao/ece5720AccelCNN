# Declare the variable
CC=gcc
CFLAGS=-c -std=gnu99

output: cnn.o read_data.o
	$(CC) cnn.o read_data.o -o output

cnn.o: cnn.c util_func.h
	$(CC) $(CFLAGS) cnn.c

read_data.o: read_data.c mnist.h
	$(CC) $(CFLAGS) read_data.c

clean:
	rm -rf *.o output