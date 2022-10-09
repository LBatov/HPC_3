#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main (int argc, char *argv[])
{

   for (int j=0; j<5 ; j++)
   {
       double r_sq;
       int hits, cnt;
       hits = 0;
       cnt = 0;
       r_sq = 0.25;
   #pragma omp parallel default(none) shared(r_sq, hits, cnt, j)

        {
            double x_coord;
            double y_coord;

            unsigned int ts_seed = time(NULL);
            unsigned int myseed = (omp_get_thread_num() + ts_seed) + j;

            srand(myseed);

            #pragma omp for reduction(+:hits,cnt)
            for(int i=0; i<1000000 ; i++)
            {

                x_coord = (double)rand()/RAND_MAX;
                y_coord = (double)rand()/RAND_MAX;

                cnt += 1;

                if (((x_coord - 0.5) * (x_coord - 0.5) + (y_coord - 0.5) *  (y_coord - 0.5) ) < 0.25)
                {
                    hits+=1;
                }

            }

        }
        printf("hits=%d cnt=%d pi=%f\n",hits,cnt, 4.0*hits/cnt);
   }
}
