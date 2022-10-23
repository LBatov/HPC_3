#include <stdlib.h>
#include <omp.h>
#include <stdio.h>

//умножение матрицы на вектор
void matpower_v(double *first_m, double *second_m, double *result,  int N)
{
    int i, j, k;
    omp_set_num_threads(8);

    #pragma omp parallel for private(i,j)
    for (i=0; i < N; i++)
        {
            result[i] = (double)0;
        }
    #pragma omp parallel for private(i,j,k) collapse(2)
    for (i=0; i < N; i++)
        for(j=0; j < N; j++)
        {
            result[i] += first_m[j * N + i] * second_m[j];
        }
}

void pagerank (double *Adj_m, int N, int iterations, double d, double *res)
{
    double *v = (double *)malloc(N * sizeof(double));
    double *M_hat = (double *)malloc(N * N* sizeof(double));
    double damping_const = (double)((1 - d) / N);
    double *aux = (double *)malloc(N * N* sizeof(double));

    int i, j, k;

    //Начальный вектор
    for (i=0; i < N; i++)
    {
        v[i] = 1.0 / N;
    }

    //Матрица с учетом Damping factor
    #pragma omp parallel for private(i,j) collapse(2)
    for (i=0; i < N; i++)
        for(j=0; j < N; j++)
            {
                M_hat[i * N + j] = Adj_m[i * N + j] * d + damping_const;
            }

     //вывод для удобного импорта в питон для броверки
     printf("hat matrix \n[[");
     for (k=0; k < N * N ;k++)
        {   if ((k % N == 0) && k > 0)
                printf("],\n[");
            printf("%f, ", M_hat[k]);

        }
     printf("]]\n \n");

     for (i=0; i < iterations; i++)
        {
            matpower_v(M_hat, v, aux, N);
            memcpy (v, aux, N * sizeof(double));
         }

     memcpy (res, v, N * sizeof(double));

     free (M_hat);
     free (aux);
     free (v);
}

void main()
{
    int i,j,k;
    int N = 34;
    int iterations = 200;
    double d = 0.85;
    double *A = (double *)malloc(N * N* sizeof(double));
    double *res = (double *)malloc(N * sizeof(double));

    //загрузка матрицы смежности
    FILE *myFile;
    myFile = fopen("adj.txt", "r");
    for (i=0;i < N * N;i++)
    {
      fscanf(myFile, "%d,", &A[i]);
    }

    //нормировка колонок матрицы смежности
    for  (i=0; i < N; i++)
    {
       double j_sum = 0.0;
       for (j=0; j < N;j++)
       {
           j_sum+=A[i * N + j];
       }
       if (j_sum > 0)
            for (j=0; j < N;j++)
               {
                   A[i * N + j] = A[i * N + j] / j_sum;
               }
        }
    //вывод матрицы смежности дляповерки
    printf("init matrix \n[[");
    for (k=0; k < N * N ;k++)
        {   if ((k % N == 0) && k > 0)
                printf("],\n[");
            printf("%f, ", A[k]);

        }
    printf("]]\n \n");

    pagerank (A,  N, iterations, d, res);

    printf("res pagerank, iterations=%d \n", iterations);
    for (k=0; k < N  ;k++)
        {   if (k % N == 0)
                printf("\n");
            printf("%f ", res[k]);
        }

    free(A);
    free(res);
 }
