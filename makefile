# Declare the variable
CC=gcc
CFLAGS=-c -Wall

output: cnn.o read_data.o
	$(CC) cnn.o read_data.o -o output

cnn.o: cnn.c
	$(CC) -c cnn.c

read_data.o: read_data.c mnist.h
	$(CC) -c read_data.c

clean:
	rm -rf *.o output