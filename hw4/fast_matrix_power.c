#include <stdlib.h>
#include <omp.h>

long matpower2(long *first_m, long *second_m, int N)
{
long* result = (long *)malloc(N * N* sizeof(long));

int i, j, k;

omp_set_num_threads(8);

#pragma omp parallel for private(i,j) collapse(2)
for (i=0; i < N; i++)
    for(j=0; j < N; j++)
    {
        result[i * N + j] = 0;

    }
#pragma omp parallel for private(i,j,k) collapse(3)
    for (i=0; i < N; i++)
        for(k=0; k < N; k++)
        {
            for(j=0; j < N; j++)
            {
                result[i * N + j] += first_m[i * N + k] * second_m[k * N + j];
            }
        }
return result;
}

long recursive_power(long *Adj_m, int N, int power)
{
    if (power == 0 )
        {
            long* e = (long *)malloc(N * N* sizeof(long));
            int i, j;
            #pragma omp for private(i,j) collapse(2)
            for (i=0; i < N; i++)
                for(j=0; j < N; j++)
                {
                    if (i == j)
                        e[i * N + j] = 1;
                    else
                        e[i * N + j] = 0;
                }
            return e;
        }
    else if (power % 2 == 0)
        return recursive_power(matpower2(Adj_m, Adj_m, N), N, power / 2);
    else
        return matpower2(Adj_m, recursive_power(matpower2(Adj_m, Adj_m, N), N, (power - 1) / 2), N);
}

void main()
{
    int N = 5;
    int power = 10;
    long *A = (long *)malloc(N * N* sizeof(long));
    long *B = (long *)malloc(N * N* sizeof(long));

     unsigned int ts_seed = time(NULL);
     srand(ts_seed);
     
    for (int i; i < N * N;i++)
        if ((double)rand()/RAND_MAX > 0.5)
            A[i] = 1;
        else
            A[i] = 0;

    printf("init matrix \n[[");
    for (int k=0; k < N * N ;k++)
        {   if ((k % N == 0) && k > 0)
                printf("],\n[");
            printf("%d, ", A[k]);

        }
    printf("]]\n \n");
    B = recursive_power(A, N, power);


    printf("res matrix, power=%d \n", power);
    for (int k2=0; k2 < N * N ;k2++)
        {   if (k2 % N == 0)
                printf("\n");
            printf("%d ", B[k2]);
        }

    free(A);
    free(B);
 }
