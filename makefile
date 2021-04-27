# Declare the variable
CC=gcc
CFLAGS=-c -std=gnu99

output: cnn.o read_data.o conv_layer.o relu_layer.o
	$(CC) cnn.o read_data.o conv_layer.o relu_layer.o -o output

cnn.o: cnn.c
	$(CC) $(CFLAGS) cnn.c

read_data.o: read_data.c
	$(CC) $(CFLAGS) read_data.c

conv_layer.o: conv_layer.c
	$(CC) $(CFLAGS) conv_layer.c

relu_layer.o: relu_layer.c
	$(CC) $(CFLAGS) relu_layer.c

clean:
	rm -rf *.o output