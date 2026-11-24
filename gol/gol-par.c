/***********************

Conway's Game of Life

Based on https://web.cs.dal.ca/~arc/teaching/CS4125/2014winter/Assignment2/Assignment2.html

************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include "mpi.h"

typedef struct {
   int rows, cols;
   int **cells;
} world;

/* keep short history since we want to detect simple cycles */
#define HISTORY 3

static world worlds[HISTORY];
static int world_iter = 0;
static int world_rows, world_cols;

static world *cur_world;

static int print_cells = 0;
static int print_world = 0;

// use fixed world or random world?
#ifdef FIXED_WORLD
static int random_world = 0;
#else
static int random_world = 1;
#endif

static char *start_world[] = {
    /* Gosper glider gun */
    /* example from https://bitstorm.org/gameoflife/ */
    "..........................................",
    "..........................................",
    "..........................................",
    "..........................................",
    "..........................................",
    "..........................................",
    "........................OO.........OO.....",
    ".......................O.O.........OO.....",
    ".OO.......OO...........OO.................",
    ".OO......O.O..............................",
    ".........OO......OO.......................",
    ".................O.O......................",
    ".................O........................",
    "....................................OO....",
    "....................................O.O...",
    "....................................O.....",
    "..........................................",
    "..........................................",
    ".........................OOO..............",
    ".........................O................",
    "..........................O...............",
    "..........................................",
};

static int rank;
static int size;
static int rows_per_rank;
static int start_row_index;
static int end_row_index;
static int process_rows;
static int last_rank;

static MPI_Status status;

static void
world_init_fixed(world *world)
{
    int **cells = world->cells;
    int row, col;

    /* use predefined start_world */

    for (row = 1; row <= world->rows; row++) {
        for (col = 1; col <= world->cols; col++) {
            if ((row <= sizeof(start_world) / sizeof(char *)) &&
                (col <= strlen(start_world[row - 1]))) {
                cells[row][col] = (start_world[row - 1][col - 1] != '.');
            } else {
                cells[row][col] = 0;
            }
        }
    }
}

static void
world_init_random(world *world)
{
    int **cells = world->cells;
    int row, col;

    // Note that rand() implementation is platform dependent.
    // At least make it reprodible on this platform by means of srand()
    srand(1);

    for (row = 1; row <= world->rows; row++) {
        for (col = 1; col <= world->cols; col++) {
            float x = rand() / ((float)RAND_MAX + 1);
            if (x < 0.5) {
                cells[row][col] = 0;
            } else {
                cells[row][col] = 1;
            }
        }
    }
}

static void
world_print(world *world)
{
    int **cells = world->cells;
    int row, col;

    for (row = 1; row <= world->rows; row++) {
        for (col = 1; col <= world->cols; col++) {
            if (cells[row][col]) {
                printf("O");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }
}

static int
world_count(world *world)
{
    int **cells = world->cells;
    int isum, reduce_isum;
    int row, col;

    isum = 0;
    for (row = start_row_index; row < end_row_index; row++) {
        for (col = 1; col <= world->cols; col++) {
            isum = isum + cells[row][col];
        }
    }

    MPI_Reduce(&isum, &reduce_isum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    return reduce_isum;
}

/* Take world wrap-around into account: */
static void
world_border_wrap(world *world)
{
    int **cells = world->cells;
    int row, col;

    /* left-right boundary conditions */
    for (row = start_row_index; row < end_row_index; row++) {
        cells[row][0] = cells[row][world->cols];
        cells[row][world->cols + 1] = cells[row][1];
    }

    /* top boundary conditions */
    if (rank == 0) {
        if (last_rank != 0) {
            MPI_Sendrecv(cells[1], world->cols+2, MPI_INT, last_rank, 0,
                cells[world->rows], world->cols+2, MPI_INT, last_rank, 0, MPI_COMM_WORLD, &status);
        }
        for (col = 0; col <= world->cols + 1; col++) {
            cells[0][col] = cells[world->rows][col];
        }
    }

    /* bottom boundary conditions */
    if (rank == last_rank) {
        if (rank != 0) {
            MPI_Sendrecv(cells[world->rows], world->cols+2, MPI_INT, 0, 0,
                cells[1], world->cols+2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        }
        for (col = 0; col <= world->cols + 1; col++) {
            cells[world->rows + 1][col] = cells[1][col];
        }
    }
}

static void
world_left_right_border_wrap(world *world)
{
    int **cells = world->cells;
    int row;

    /* left-right boundary conditions */
    for (row = start_row_index; row < end_row_index; row++) {
        cells[row][0] = cells[row][world->cols];
        cells[row][world->cols + 1] = cells[row][1];
    }
}

// update board for next timestep
// rows/cols params are the base rows/cols
// excluding the surrounding 1-cell wraparound border
static void
world_timestep(world *old, world *new)
{
    int **cells = old->cells;
    int row, col;

    // comunication with up-down board rows
    if (rank > 0 && rank <= last_rank) {
        MPI_Sendrecv(cells[start_row_index], old->cols+2, MPI_INT, rank-1, 0,
            cells[start_row_index-1], old->cols+2, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);
    }
    if (rank < last_rank) {
        MPI_Sendrecv(cells[end_row_index-1], old->cols+2, MPI_INT, rank+1, 0,
            cells[end_row_index], old->cols+2, MPI_INT, rank+1, 0, MPI_COMM_WORLD, &status);
    }

    // update board
    for (row = start_row_index; row < end_row_index; row++) {
        for (col = 1; col <= new->cols; col++) {
            int row_m, row_p, col_m, col_p, nsum;
            int newval;

            // sum surrounding cells
            row_m = row - 1;
            row_p = row + 1;
            col_m = col - 1;
            col_p = col + 1;

            nsum = cells[row_p][col_m] + cells[row_p][col] + cells[row_p][col_p]
                 + cells[row  ][col_m]                     + cells[row  ][col_p]
                 + cells[row_m][col_m] + cells[row_m][col] + cells[row_m][col_p];

            switch (nsum) {
            case 3:
                // a new cell is born
                newval = 1;
                break;
            case 2:
                // the cell, if any, survives
                newval = cells[row][col];
                break;
            default:
                // the cell, if any, dies
                newval = 0;
                break;
            }

            new->cells[row][col] = newval;
        }
    }
}

static int world_check_cycles(world *cur_world, int iter) {
    int i;
    int comps[HISTORY];
    int reduce_comps[HISTORY];
    int res = 0;

    /* This version re-applies the border wraps so they are consistent with
    * the respective world states, and we can just compare the full world arrays.
    */
    world_left_right_border_wrap(cur_world);

    for (i = iter - 1; i >= 0 && i > iter - HISTORY; i--) {
        world *prev_world = &worlds[i % HISTORY];

        world_left_right_border_wrap(prev_world);

        if (rank <= last_rank) {
            comps[i % HISTORY] = memcmp(
                &cur_world->cells[start_row_index][0],
                &prev_world->cells[start_row_index][0],
                (end_row_index - start_row_index) * (world_cols + 2) * sizeof(int));
        } else {
            comps[i % HISTORY] = 0;
        }
    }

    MPI_Reduce(comps, reduce_comps, HISTORY, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (i = iter - 1; i >= 0 && i > iter - HISTORY; i--) {
            if (reduce_comps[i % HISTORY] == 0) {
                printf("world iteration %d is equal to iteration %d\n", iter, i);
                res = 1;
            }
        }
    }

    MPI_Bcast(&res, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return res;
}

static int **
alloc_2d_int_array(int nrows, int ncolumns)
{
    int **array;
    int row;

    /* version that keeps the 2d data contiguous, can help caching and slicing across dimensions */
    array = malloc(nrows * sizeof(int *));
    if (array == NULL) {
       fprintf(stderr, "out of memory\n");
       exit(1);
    }

    array[0] = malloc(nrows * ncolumns * sizeof(int));
    if (array[0] == NULL) {
       fprintf(stderr, "out of memory\n");
       exit(1);
    }

    /* memory layout is row-major */
    for (row = 1; row < nrows; row++) {
        array[row] = array[0] + row * ncolumns;
    }

    return array;
}

static double
time_secs(void)
{
    struct timeval tv;

    if (gettimeofday(&tv, 0) != 0) {
        fprintf(stderr, "could not do timing\n");
        exit(1);
    }

    return tv.tv_sec + (tv.tv_usec / 1000000.0);
}

static void
mpi_init(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int flag;
    MPI_Initialized(&flag);
    if (flag == 0) {
        fprintf(stderr, "mpi initialize failed\n");
        exit(0);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
}

int
main(int argc, char *argv[])
{
    int h, nsteps;
    int cnt;
    world w;
    double start_time, end_time, elapsed_time;

    mpi_init(argc, argv);

    /* Get Parameters */
    if (argc != 6) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s rows cols steps worldstep cellstep\n", argv[0]);
        exit(1);
    }

    world_rows = atoi(argv[1]);
    world_cols = atoi(argv[2]);
    nsteps = atoi(argv[3]);
    print_world = atoi(argv[4]);
    print_cells = atoi(argv[5]);

    rows_per_rank = (world_rows + size - 1)/size;
    start_row_index = rows_per_rank * rank + 1;
    if (start_row_index > world_rows) {
        start_row_index = end_row_index = world_rows + 1;
    } else {
        if (start_row_index + rows_per_rank < world_rows + 1) {
            end_row_index = start_row_index + rows_per_rank;
        } else {
            end_row_index = world_rows + 1;
        }
    }
    process_rows = end_row_index - start_row_index;
    last_rank = (world_rows + rows_per_rank - 1) / rows_per_rank - 1;

    /* initialize worlds, when allocating arrays, add 2 for ghost cells in both directorions */
    for (h = 0; h < HISTORY; h++) {
        worlds[h].rows = world_rows;
        worlds[h].cols = world_cols;
        worlds[h].cells = alloc_2d_int_array(rows_per_rank*size + 2, world_cols + 2);
    }

    w.rows = world_rows;
    w.cols = world_cols;
    w.cells = alloc_2d_int_array(rows_per_rank*size + 2, world_cols + 2);

    if (rank == 0) {
        if (random_world) {
            world_init_random(&w);
        } else {
            world_init_fixed(&w);
        }
    }

    /*  initialize board */
    cur_world = &worlds[world_iter % HISTORY];

    /*  distribution data */
    MPI_Scatter(w.cells[1], rows_per_rank*(world_cols+2), MPI_INT,
                cur_world->cells[start_row_index], rows_per_rank*(world_cols+2), MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0 && print_world > 0) {
        printf("\ninitial world:\n\n");
        world_print(&w);
    }

    start_time = time_secs();

    /*  time steps */
    for (world_iter = 1; world_iter < nsteps; world_iter++) {
        world *next_world;
        int cycle;

        next_world = &worlds[world_iter % HISTORY];

        world_border_wrap(cur_world);
        world_timestep(cur_world, next_world);
        cur_world = next_world;

        cycle = world_check_cycles(cur_world, world_iter);

        if (print_cells > 0 && (world_iter % print_cells) == (print_cells - 1)) {
            cnt = world_count(cur_world);
            if (rank == 0) {
                printf("%d: %d live cells\n", world_iter, cnt);
            }
        }

        if (print_world > 0 && (cycle || (world_iter % print_world) == (print_world - 1))) {
            MPI_Gather(cur_world->cells[start_row_index], rows_per_rank * (world_cols+2), MPI_INT,
                        w.cells[1], rows_per_rank * (world_cols+2), MPI_INT, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("\nat time step %d:\n\n", world_iter);
                world_print(&w);
            }
        }

        if (cycle) {
            break;
        }
    }

    end_time = time_secs();
    elapsed_time = end_time - start_time;
    cnt = world_count(cur_world);
    if (rank == 0) {
        /*  Iterations are done; sum the number of live cells */
        printf("Number of live cells = %d\n", cnt);
        fprintf(stderr, "Game of Life took %10.3f seconds\n", elapsed_time);
    }

    MPI_Finalize();

    return 0;
}