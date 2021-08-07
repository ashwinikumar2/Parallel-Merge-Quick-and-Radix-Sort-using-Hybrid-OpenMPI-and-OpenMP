all:
	mpic++ -fopenmp -c -shared -o libpsort.so -fPIC sort.cpp

clean:
	rm -f ./exe ./libpsort.a ./sort.o