#pragma once

#define LOADSIZE 40

class pSort {

public:
   typedef enum {
   	ONE = 1,
   	TWO = 2,
   	THREE = 3
   }  SortType;

   typedef struct {
      long long key;
      char payload[LOADSIZE];
   } dataType;

   void init(); // Initialize called before sorting
   void close(); // Initialize called after sorting/searching
   int sort(dataType **data, int *ndata, SortType sorter); // Return 0 on success, 1 on failure
   char *search(dataType *data, int ndata, long long key); // Return the payload pointer
};
