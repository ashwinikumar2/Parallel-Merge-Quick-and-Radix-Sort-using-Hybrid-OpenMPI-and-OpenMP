#include <mpi.h>
#include "sort.h"
#include<iostream>
#include <vector>
#include <cmath>
#include <omp.h> 
#include <bits/stdc++.h>
using namespace std;

MPI_Datatype newtype;           //A Custom data type representing psort::datatype

void pSort::init()              //Used to initialize the program
{
    MPI_Init(NULL, NULL);
    int size_of_type;
    MPI_Type_size(MPI_LONG_LONG, &size_of_type);
    MPI_Aint displacements[2]  = {0,size_of_type};
    int block_lengths[2]  = {1, LOADSIZE};
    MPI_Datatype type_of_blocks[2] = {MPI_LONG_LONG,MPI_CHAR};
   
    MPI_Type_create_struct(2,block_lengths,displacements,type_of_blocks,&newtype);      //creating a new struct for psort::datatype consisting of two blocks of lengths 1 and LOADSIZE
    MPI_Type_commit(&newtype);                      //commiting the new custom data type named as 'newtype'
}

void pSort::close()             //Used to close the MPI program
{
    MPI_Finalize();
}

pSort::dataType* merge_arrays(pSort::dataType *left, pSort::dataType *right, int leftsize, int rightsize)           //used to merge two arrays as recieved. It does not change the order of elements just appends the right arrau ahead of left array
{
    pSort::dataType *output=new pSort::dataType[leftsize+rightsize];
    for(int i=0;i<leftsize;i++)
    {
        output[i]= left[i];

    }

    for(int i=0;i<rightsize;i++)
    {
        output[i+leftsize]= right[i];
        
    }

    return output;
}

pSort::dataType* sequential_quick_sort(pSort::dataType *data, int ndata, int low, int high)                         //serial quick sort implementing the usual quick sort algorithm by finding the partition index and 
{                                                                                                       //then recursively use this on left and right portions seperated by partition index
    if(low<high)
    {
        // pSort::dataType *temp=data;

        pSort::dataType pivot = data[high];                         //taking pivot to be the value of key of last element in the recieved array  
        int i = (low - 1); 
      
        for (int j=low;j<high;j++)  
        {  
            if (data[j].key < pivot.key)  
            {  
                i++; 
                pSort::dataType x=data[i];
                data[i]=data[j];                                    //swapping values of array elements at indexes i and j
                data[j]=x; 
            }  
        }  
        pSort::dataType x=data[i+1];
        data[i+1]=data[high];                                       //swapping values of array elements at indexes i+1 and high
        data[high]=x;

        int pi=i+1;
        data= sequential_quick_sort(data, ndata, low, pi-1);
        data=sequential_quick_sort(data, ndata, pi+1, high);
        return data;
        
    }
    return data;
}
pSort::dataType* mp_quick_sort(pSort::dataType *data, int ndata, int low, int high)
{
    if(high-low+1<=1<<16){                                 //if size of quick sort is less than 10 then use sequential quick sort
        data=sequential_quick_sort(data, ndata, low, high);
        return data;
    }
    if(low<high)                                        //if low is less than high then only perform the computation for quick sort
    {
        // pSort::dataType *temp=data;

        pSort::dataType pivot = data[high];                         //taking pivot to be the value of key of last element in the recieved array  
        int i = (low - 1); 
      
        for (int j=low;j<high;j++)  
        {  
            if (data[j].key < pivot.key)  
            {  
                i++; 
                pSort::dataType x=data[i];
                data[i]=data[j];                                    //swapping values of array elements at indexes i and j
                data[j]=x; 
            }  
        }  
        pSort::dataType x=data[i+1];
        data[i+1]=data[high];                                       //swapping values of array elements at indexes i+1 and high
        data[high]=x;

        int pi=i+1;
        #pragma omp parallel sections                                      //start parallel computation using sections
        {
                #pragma omp section                                                //if it is the first section then call recursively on left subarray
                {
                    data=mp_quick_sort(data, ndata, low, pi-1);
                }
                #pragma omp section                                                    //else call recursively on right subarray
                {
                    data=mp_quick_sort(data, ndata, pi+1, high);
                }
        }
        return data;
        
    }
    return data;
}

pSort::dataType* quick_sort(pSort::dataType *data, int &ndata, MPI_Comm comm)                                       //parallel version of quick sort
{
    pSort::dataType *temp=data;
    int my_rank,ndata_new;
    int recv_count,  i;
    MPI_Status status;
    int numProcess;    
    MPI_Comm_size(comm, &numProcess);
    MPI_Comm_rank(comm, &my_rank);
    pSort::dataType pivot1;             
    if(numProcess==1){                                                                                  //if number of processors are 1 in this communicator then apply serial quick sort                                      
        if(ndata!=0)
        data=mp_quick_sort(data, ndata, 0, ndata-1); 

        for(int i=0;i<ndata;i++)                                                    //used for setting the pointer values of what was obtained as data at initialization of this function
        {
            temp[i]=data[i];
        }
        return data;
    }

    int low=0,high=numProcess-1;
    if (my_rank==(numProcess-1))                                                    //if this is the last processor 
    {
        pSort::dataType pivot=data[ndata/2];
        MPI_Bcast(&pivot, 1,newtype,my_rank, comm);                                 //then broadcast to every other processor the pivot value i.e. at ndata/2 index 
        int x1=0;
        int left_size=0,right_size=0;
        for(int i=0;i<ndata;i++)
        {
            if(data[i].key<=pivot.key)
                left_size++;
            else
                right_size++;
        }

        pSort::dataType *left=new pSort::dataType[left_size];
        pSort::dataType *right= new pSort::dataType[right_size];
        int left_index=0,right_index=0;
        for(int i=0;i<ndata;i++)
        {
            if(data[i].key<=pivot.key){
                left[left_index]=data[i];
                left_index++;
          }
            else{
                right[right_index]=data[i];
                right_index++;
          }
        }
        MPI_Send( left, left_size, newtype, x1, 10, comm); 
        MPI_Status statusx;
        MPI_Probe(x1,10,comm,&statusx);
        int recv_count;
        MPI_Get_count(&statusx, newtype, &recv_count);
        pSort::dataType *recieving_high_list_from_partner=new pSort::dataType[recv_count];
        
        MPI_Recv( recieving_high_list_from_partner, recv_count, newtype, x1,  10, comm, &status);     //recieve the values greater than pivot from the partner process which belongs to lower half list of processors         
        data= merge_arrays(right, recieving_high_list_from_partner, right_size, recv_count);

        ndata_new=right_size+recv_count;
        pivot1=pivot;

        if(my_rank==(numProcess/2)+1 && numProcess%2!=0)
        {

            MPI_Status statusx1;
            MPI_Probe(numProcess/2,10,comm,&statusx1);
            int recv_count1;
            MPI_Get_count(&statusx1, newtype, &recv_count);
            pSort::dataType *recieving_high_list_from_partner1=new pSort::dataType[recv_count];
            MPI_Recv( recieving_high_list_from_partner1, recv_count, newtype, numProcess/2,  MPI_ANY_TAG, comm, &status); //recieve the values greater than pivot from the process at rank= number_of_processors/2 which does not have any partner process

            data= merge_arrays(recieving_high_list_from_partner1,data, recv_count, ndata_new);
            ndata_new+=recv_count;
            delete[] recieving_high_list_from_partner1;
        }

        // left1.clear();
        // right1.clear();
        delete[] right;
        delete[] recieving_high_list_from_partner;
        
    }
    //reorganising data from every processor such that upper processors contain data greater than pivot and lower processors contains data less than or equal to pivot
    else 
    {
        if(my_rank>((numProcess-1)/2) )     //upper half listed processor send data to partner processor in lower half and recieve high list data(data greater than pivot) from partner processor
        {
            pSort::dataType pivot;
            MPI_Bcast(&pivot, 1,newtype, numProcess-1, comm);                   //calling broadcast so that this processor can recieve the pivot value from last processor
            int left_size=0,right_size=0;
            for(int i=0;i<ndata;i++)
            {
                if(data[i].key<=pivot.key)
                    left_size++;
                else
                    right_size++;
            }

            pSort::dataType *left= new pSort::dataType[left_size];
            pSort::dataType *right= new pSort::dataType[right_size];
            int left_index=0,right_index=0;
            for(int i=0;i<ndata;i++)
            {
                if(data[i].key<=pivot.key){
                    left[left_index]=data[i];
                    left_index++;
              }
                else{
                    right[right_index]=data[i];
                    right_index++;
              }
            }
            data=new pSort::dataType[right_size];
            for(int i=0;i<right_size;i++)
                data[i]=right[i];
            int x1=numProcess-1-my_rank;

            MPI_Send( left, left_size, newtype, x1, 10, comm);                   //send the values less than pivot to the partner process which belongs to lower half list of processors

            MPI_Status statusx;
            MPI_Probe(x1,10,comm,&statusx);
            int recv_count;
            MPI_Get_count(&statusx, newtype, &recv_count);
            pSort::dataType *recieving_high_list_from_partner=new pSort::dataType[recv_count];
            MPI_Recv( recieving_high_list_from_partner, recv_count, newtype, x1,  MPI_ANY_TAG, comm, &status);    //recieve the values greater than pivot from the partner process which belongs to lower half list of processors         

            data= merge_arrays(data, recieving_high_list_from_partner, right_size,recv_count);
            ndata_new=right_size+recv_count;
            pivot1=pivot;

            if(my_rank==(numProcess/2)+1 && numProcess%2!=0)                        
            {
                MPI_Status statusx1;
                MPI_Probe(numProcess/2,10,comm,&statusx1);
                int recv_count1;
                MPI_Get_count(&statusx1, newtype, &recv_count);
                pSort::dataType *recieving_high_list_from_partner1=new pSort::dataType[recv_count];
                MPI_Recv( recieving_high_list_from_partner1, recv_count, newtype, numProcess/2,  MPI_ANY_TAG, comm, &status);           //recieve the values greater than pivot from the process at rank= number_of_processors/2 which does not have any partner process

                data= merge_arrays(recieving_high_list_from_partner1,data, recv_count, ndata_new);
                ndata_new+=recv_count;
                delete[] recieving_high_list_from_partner1;
            }
            // left1.clear();
            // right1.clear();
            delete[] right;
            delete[] recieving_high_list_from_partner;
          

        }
        else                        //lower half processor receive low list data from partner processor (in upper half of list of processors) and sends a high list of data to partner processor
        {
            pSort::dataType pivot;
            MPI_Bcast(&pivot, 1,newtype, numProcess-1, comm);                       //calling broadcast so that this processor can recieve the pivot value from last processor
            
            int recieving_rank=(numProcess-1-my_rank);
            
            if(my_rank==numProcess/2 && numProcess%2!=0)                            //if the number of processors are odd which leaves the processor at rank=number_of_processors/2 with no partner processor
            {
                int left_size=0,right_size=0;
                for(int i=0;i<ndata;i++)
                {
                    if(data[i].key<=pivot.key)
                        left_size++;
                    else
                        right_size++;
                }

                pSort::dataType *left= new pSort::dataType[left_size];
                pSort::dataType *right= new pSort::dataType[right_size];
                int left_index=0,right_index=0;
                for(int i=0;i<ndata;i++)
                {
                    if(data[i].key<=pivot.key){
                        left[left_index]=data[i];
                        left_index++;
                  }
                    else{
                        right[right_index]=data[i];
                        right_index++;
                  }
                }
                data=new pSort::dataType[left_size];
                for(int i=0;i<left_size;i++)
                {
                    data[i]=left[i];
                }
                ndata_new=left_size;
                MPI_Send( right, right_size, newtype, (my_rank+1)%numProcess, 10, comm);         //send values greaer than pivot to the next processor in this special case
                
                // left1.clear();
                // right1.clear();
                
            }
            else
            {
                MPI_Status statusx;
                MPI_Probe(recieving_rank,10,comm,&statusx);
                int recv_count;
                MPI_Get_count(&statusx, newtype, &recv_count);
                
                pSort::dataType *numbertoreceive=new pSort::dataType[recv_count];
                MPI_Recv( numbertoreceive, recv_count, newtype, recieving_rank,  MPI_ANY_TAG, comm, &status);               //recieve the values less than pivot from the partner process in the upper half 
                    
                int left_size=0,right_size=0;
                for(int i=0;i<ndata;i++)
                {
                    if(data[i].key<=pivot.key)
                        left_size++;
                    else
                        right_size++;
                }

                pSort::dataType *left= new pSort::dataType[left_size];
                pSort::dataType *right= new pSort::dataType[right_size];
                int left_index=0,right_index=0;
                for(int i=0;i<ndata;i++)
                {
                    if(data[i].key<=pivot.key){
                        left[left_index]=data[i];
                        left_index++;
                  }
                    else{
                        right[right_index]=data[i];
                        right_index++;
                  }
                }
                data=merge_arrays(left, numbertoreceive, left_size,recv_count);
                ndata_new=left_size+recv_count;
                
                MPI_Send( right, right_size, newtype, recieving_rank, 10, comm);                                 //send values greaer than pivot to the partner processor in the upper half
                // left1.clear();
                // right1.clear();
                free(right);
                delete[] left;
                delete[] numbertoreceive;

            }

            pivot1=pivot;
            //free(array);
        }
    }

    MPI_Barrier(comm);
    int color = my_rank/((numProcess+1)>>1);
    MPI_Comm row_comm;
    
    MPI_Comm_split(comm, color,my_rank, &row_comm);                                     //split the current communicator to 2 communicators
    data=quick_sort(data,ndata_new,row_comm);                                                //call recursion on the new communicators formed
    
    ndata=ndata_new;
    return data;  
}
pSort::dataType* merge(pSort::dataType *arr, pSort::dataType *arr1 , int l, int m, int r)                       //used to merge two arrays by comparing the first elements(in iteration) of left portion as compared to right portion
{                                                                                                               //or we can say the usual merge used in serial merge sort algorithm    

    for(int i = 0; i < m-l+1; i++)
    {    
        arr1[i] = arr[l + i];
    }
    for(int j = 0; j < r-m; j++)
    {
        arr1[j+m-l+1] = arr[m + 1 + j];
    }
    int i = 0,j=0,k=l; 
    
    while (i < m - l + 1 && j < r - m)                  //while we are iterating in the left portion of arr i.e. from 0 to m-l and iterating in the right portion of arr i.e. m-l+1 to r-m
    {
        if (arr1[i].key <= arr1[j+m-l+1].key)                       
        {
            arr[k] = arr1[i];
            i=i+1;
        }
        else
        {
            arr[k] = arr1[j+m-l+1];
            j=j+1;
        }
        k=k+1;
    }

    while (i < m - l + 1) 
    {
        arr[k] = arr1[i];
        i=i+1;
        k=k+1;
    }
    while (j < r - m)
    {
        arr[k] = arr1[j+m-l+1];
        j=j+1;
        k=k+1;
    }

    return arr;                                         
}
pSort::dataType* merge_special(pSort::dataType *arr, pSort::dataType *arr1,  pSort::dataType *arr2, int ndata1, int ndata2)                       //used to merge two arrays by comparing the first elements(in iteration) of left portion as compared to right portion
{                                                                                                               //or we can say the usual merge used in serial merge sort algorithm    
    // arr1=new pSort::dataType[ndata1+ndata2];
    for(int i=0;i<ndata2+ndata1;i++)
    {
        if(i<ndata1)
        arr1[i]=arr[i];
        else
            arr1[i]=arr2[i-ndata1];
    }
    arr=new pSort::dataType[ndata1+ndata2];
    for(int i=0;i<ndata1+ndata2;i++)
    {
        arr[i]=arr1[i];
    }
    int l=0,m=ndata1-1,r=ndata2+ndata1-1;
    int i = 0,j=0,k=l; 
    
    while (i < m - l + 1 && j < r - m)                  //while we are iterating in the left portion of arr i.e. from 0 to m-l and iterating in the right portion of arr i.e. m-l+1 to r-m
    {
        if (arr1[i].key <= arr1[j+m-l+1].key)                       
        {
            arr[k] = arr1[i];
            i=i+1;
        }
        else
        {
            arr[k] = arr1[j+m-l+1];
            j=j+1;
        }
        k=k+1;
    }

    while (i < m - l + 1) 
    {
        arr[k] = arr1[i];
        i=i+1;
        k=k+1;
    }
    while (j < r - m)
    {
        arr[k] = arr1[j+m-l+1];
        j=j+1;
        k=k+1;
    }

    return arr;                                        
}
pSort::dataType* mergeSort(pSort::dataType *arr, int l, int r)                                                  //serial merge sort used for doing merge sort serially on an array by the usual merge sort algorithm
{
    if (l < r)
    {   pSort:: dataType *temp=arr;

        int m = (l+r) / 2;
        pSort::dataType *arr1=new pSort::dataType[r-l+1];
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        pSort::dataType *output=merge(arr,arr1, l, m, r);
        for (int i = l; i < r; i++)                                                                 //used for setting the pointer values of what was obtained as data at initialization of this function
        {
            /* code */
            temp[i]=output[i];
        }
        return output;
    }
    return arr;
}

pSort::dataType* mp_merge_sort(pSort::dataType *data, int ndata, int low, int high)
{
    if(high-low+1<=1<<16){                                 //if size of quick sort is less than 10 then use sequential quick sort
        data=mergeSort(data, low, high);
        return data;
    }
    if(low<high)                                        //if low is less than high then only perform the computation for quick sort
    {
        #pragma omp parallel sections                                      //start parallel computation using sections
        {
                #pragma omp section                                                //if it is the first section then call recursively on left subarray
                {
                    data=mp_merge_sort(data, ndata, low, (low+high)/2);
                }
                #pragma omp section                                                    //else call recursively on right subarray
                {
                    data=mp_merge_sort(data, ndata, (low+high)/2 +1, high);
                }
        }
        pSort::dataType *arr1=new pSort::dataType[high-low+1];          //a dummy array for passing in merge function
        data=merge(data, arr1,low, (high+low)/2, high); 
        return data;
        
    }
    
    return data;
}

pSort::dataType* merge_sort(pSort::dataType *data, int ndata)
{
    data=mp_merge_sort(data,ndata,0,ndata-1);
    int numProcess;  
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);

    for(int i=0;i<numProcess;i++)
    {
        for(int j=i;j<numProcess;j++)
        {
            if(i!=j)
            {
                if(my_rank==j){
                    MPI_Send( data, ndata, newtype, i, 10, MPI_COMM_WORLD);
                    MPI_Status statusx;
                    MPI_Probe(i,10,MPI_COMM_WORLD,&statusx);
                    int recv_count;
                    MPI_Get_count(&statusx, newtype, &recv_count);
                    pSort::dataType *data2=new pSort::dataType[recv_count];
                    
                    MPI_Recv( data2, recv_count, newtype, i,  10, MPI_COMM_WORLD, &statusx);     //recieve the values greater than pivot from the partner process which belongs to lower half list of processors         
                    data=new pSort::dataType[recv_count];
                    for(int i1=0;i1<recv_count;i1++)
                        data[i1]=data2[i1];
                    delete[] data2;
                }
                if(my_rank==i)
                {
                    MPI_Status statusx;
                    MPI_Probe(j,10,MPI_COMM_WORLD,&statusx);
                    int recv_count;
                    MPI_Get_count(&statusx, newtype, &recv_count);
                    pSort::dataType *data2=new pSort::dataType[recv_count];
                    
                    MPI_Recv( data2, recv_count, newtype, j,  10, MPI_COMM_WORLD, &statusx);     //recieve the values greater than pivot from the partner process which belongs to lower half list of processors         
                    pSort::dataType* arr1=new pSort::dataType[ndata+recv_count];
                    arr1=merge_special(data,arr1,data2,ndata,recv_count);
                    for(int i1=0;i1<ndata;i1++)
                    {
                        data[i1]=arr1[i1];
                    }

                    pSort::dataType* right=new pSort::dataType[recv_count];
                    for(int i1=0;i1<recv_count;i1++)
                    {
                        right[i1]=arr1[i1+ndata];
                    }
                    MPI_Send( right, recv_count, newtype, j, 10, MPI_COMM_WORLD);

                    delete[] data2;
                    delete[] arr1;
                    delete[] right;
                }
            }
        }
    }
    
    return data;

}

// int pSort::sort(dataType *data, int ndata, SortType sorter)
int pSort::sort(pSort::dataType **data, int *ndata, SortType sorter)
{
    pSort::dataType** temp=data;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    pSort::dataType* data2=*data;
    if(sorter== ONE){
        *data=quick_sort(data2, *ndata, MPI_COMM_WORLD);
    }
    else if (sorter== TWO)
        *data=merge_sort(data2, *ndata);
    else 
    *data=quick_sort(data2, *ndata, MPI_COMM_WORLD); 
    const int size=*ndata;
    pSort::dataType* arr1=new pSort::dataType[2*size];
    return 0; 
}

char *pSort::search(pSort::dataType *data, int ndata, long long key)
{
    int numProcess;  
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);

    pSort::dataType *numbers=new pSort::dataType[2];
    if(ndata!=0){
    numbers[0]=data[0];
    numbers[1]=data[ndata-1];
    }
    else
    {
        numbers[0].key=LONG_LONG_MAX;
        numbers[1].key=LONG_LONG_MAX;
    }
    int process1=-1;
    if (my_rank == 0)                                                                               //if we are at the first processor(i.e. rank 0) 
    {                                                                                               
        int *rcounts =new int[numProcess];
        for(int i=0;i<numProcess;i++)
        {
            rcounts[i]=2;
        }
        int count_of_collect=2*numProcess;
        int *displs=new int[numProcess];                    //displacement array
        for(int i=0;i<numProcess;i++)
        {
            
            displs[i]=0;
        }
        for(int i=0;i<numProcess;i++)
        {
            if(i==0)
                displs[i]=0;
            else{
            displs[i]=displs[i-1]+rcounts[i-1];
            }
        }
        pSort::dataType *collect=(pSort::dataType *)malloc(count_of_collect*sizeof(pSort::dataType));           //array used for collecting the elements sent by every processor to 0th processor

        MPI_Gatherv(numbers, 2, newtype, collect, rcounts, displs, newtype, 0, MPI_COMM_WORLD); 
        
        int process=-1;
        for(int i=0;i<count_of_collect;i+=2)
        {
            if((collect[i]).key<=key && key<=(collect[i+1]).key)
            {
                process=i/2;
                break;
            }
        }
        process1=process;
        MPI_Bcast(&process, 1,MPI_INT,my_rank, MPI_COMM_WORLD);
        delete[] rcounts;
        delete[] displs;
        free(collect);

    }
    else
    {
        MPI_Gatherv(numbers, 2, newtype, NULL, NULL, 0, newtype, 0, MPI_COMM_WORLD); 
        MPI_Bcast(&process1, 1,MPI_INT,0, MPI_COMM_WORLD);
    }
    if(my_rank==process1)
    {   
        bool found=false;
        pSort::dataType element;
        for(int i=0;i<ndata;i++)
        {
            if(data[i].key==key)
            {
                found=true;
                element=data[i];
                break;
            }
        }
        char *x1;
        MPI_Bcast(&element, 1,newtype,my_rank, MPI_COMM_WORLD);
        char *x=new char[LOADSIZE];
        for(int i=0;i<LOADSIZE;i++)
        {
            x[i]=element.payload[i];    
        } 
        return x;

    }
    else
    {
        char *x1;
        pSort::dataType element;
        MPI_Bcast(&element, 1,newtype,process1, MPI_COMM_WORLD);
        char *x=new char[LOADSIZE];
        for(int i=0;i<LOADSIZE;i++)
        {
            x[i]=element.payload[i];    
        } 
        return  x;
    }
    return NULL;
}