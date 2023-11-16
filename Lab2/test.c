/*
 * test.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 * 
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>

#include "test.h"
#include "stack.h"
#include "non_blocking.h"



#define test_run(test)\
  printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, #test);\
  test_setup();\
  if(test())\
  {\
    printf("passed\n");\
  }\
  else\
  {\
    printf("failed\n");\
  }\
  test_teardown();

/* Helper function for measurement */
double timediff(struct timespec *begin, struct timespec *end)
{
	double sec = 0.0, nsec = 0.0;
   if ((end->tv_nsec - begin->tv_nsec) < 0)
   {
      sec  = (double)(end->tv_sec  - begin->tv_sec  - 1);
      nsec = (double)(end->tv_nsec - begin->tv_nsec + 1000000000);
   } else
   {
      sec  = (double)(end->tv_sec  - begin->tv_sec );
      nsec = (double)(end->tv_nsec - begin->tv_nsec);
   }
   return sec + nsec / 1E9;
}

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

#ifndef NDEBUG
int
assert_fun(int expr, const char *str, const char *file, const char* function, size_t line)
{
	if(!(expr))
	{
		fprintf(stderr, "[%s:%s:%zu][ERROR] Assertion failure: %s\n", file, function, line, str);
		abort();
		// If some hack disables abort above
		return 0;
	}
	else
		return 1;
}
#endif

stack_t *stack;
data_t data;
stack_t *pool;
_Atomic int atomic_counter = 0;

node_t* get_node(stack_t *pool){
  node_t *node;
  if(--atomic_counter>0){
    printf("reusing memory \n");
    node = stack_pop(pool);
  }
  else{
    node = malloc(sizeof(node_t));
  }
  return node;
}

void add_to_pool(stack_t *pool, node_t *node){
  ++atomic_counter;
  stack_push(-1, pool, node);
}


#if MEASURE != 0
struct stack_measure_arg
{
  int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;

#if MEASURE == 1
void*
stack_measure_pop(void* arg)
  {
    stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
    int i;

    clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
    for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
      {
        // See how fast your implementation can pop MAX_PUSH_POP elements in parallel
        stack_pop(stack);
      }
    clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);


    return NULL;
  }
#elif MEASURE == 2
void*
stack_measure_push(void* arg)
{
  stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
  int i;

  clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
  node_t *test;
  for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
    {
        //test = malloc(sizeof(test));
        stack_push(i, stack, get_node(pool));
    }
  clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);


  return NULL;
}
#endif
#endif

/* A bunch of optional (but useful if implemented) unit tests for your stack */
void
test_init()
{
  // Initialize your test batch
}

void
test_setup()
{
  // Allocate and initialize your test stack before each test
  data = DATA_VALUE;


  stack = stack_init();
  pool = stack_init();
  //node_t *node_a = malloc(sizeof(node_t));
  // node_a->val = 10;
  // node_a->next = NULL;

  // Reset explicitely all members to a well-known initial value
  // For instance (to be deleted as your stack design progresses):
  //stack->head = node_a;
}

void
test_teardown()
{
  // Do not forget to free your stacks after each test
  // to avoid memory leaks
  free(stack);
}

void
test_finalize()
{
  // Destroy properly your test batch
}


void add_to_pool_aba(stack_t *pool, node_t *node){
  stack_push(-1, pool, node);
  node_t *A;
  while(node->next!=NULL){
    A = node->next;
    node->next = A->next;
    A->next = node;
    pool->head = A;
  }
}

int
test_push_safe()
{
  // Make sure your stack remains in a good state with expected content when
  // several threads push concurrently to it
  // Do some work
  stack_push(1, stack, get_node(pool));
  stack_push(2, stack, get_node(pool));
  stack_push(5, stack, get_node(pool));
  // check if the stack is in a consistent state
  int res = assert(stack_check(stack));

  // check other properties expected after a push operation
  // (this is to be updated as your stack design progresses)
  // Now, the test succeeds
  return res && assert(stack->head->next != NULL);
}

int
test_pop_safe()
{
  stack_push(1, stack, get_node(pool));
  stack_push(2, stack, get_node(pool));
  
  stack_push(5, stack, get_node(pool));
  add_to_pool(pool, stack_pop(stack));
  add_to_pool(pool, stack_pop(stack));
  
  stack_push(5, stack, get_node(pool));
  
  add_to_pool(pool, stack_pop(stack));
  add_to_pool(pool, stack_pop(stack));
  int res = assert(stack->head == NULL);

  // Same as the test above for parallel pop operation
  // For now, this test always fails
  return res;
}


#if MEASURE == 0
// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 3
pthread_mutex_t lock0, lock1, lock2;

int thread0(){
  node_t *node_to_pop, *next;
  do{
    node_to_pop = stack->head;
    next = stack->head->next;
    printf("1 \n");
    pthread_mutex_unlock(&lock1);
    pthread_mutex_lock(&lock0);
    printf("5 \n");
    printf("pointer %p: \n", node_to_pop);
    printf("pointer %p: \n", stack->head);
  }while(node_to_pop != (node_t*)cas((size_t*)&stack->head, (size_t)node_to_pop, (size_t)next));
}

int thread1(){
  pthread_mutex_lock(&lock1);
  add_to_pool_aba(pool,stack_pop(stack));
  printf("2 \n");
  pthread_mutex_unlock(&lock2);
  pthread_mutex_lock(&lock1);
  printf("4 \n");
  stack_push(3, stack, get_node(pool));
  pthread_mutex_unlock(&lock0);
}

int thread2(){
  pthread_mutex_lock(&lock2);
  printf("3 \n");
  add_to_pool_aba(pool,stack_pop(stack));
  pthread_mutex_unlock(&lock1);
}

int
test_aba()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  int success, aba_detected = 0;
  // Write here a test for the ABA problem
  pthread_attr_t attr;
  pthread_t threads[ABA_NB_THREADS];
  
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutex_init(&lock0, NULL);
  pthread_mutex_init(&lock1, NULL);
  pthread_mutex_init(&lock2, NULL);

  pthread_mutex_lock(&lock0);
  pthread_mutex_lock(&lock1);
  pthread_mutex_lock(&lock2);

  node_t *A = get_node(pool);
  node_t *B = get_node(pool);
  node_t *C = get_node(pool);
  printf("pointer A %p: \n", A);
  printf("pointer B %p: \n", B);
  printf("pointer C %p: \n", C);

  stack_push(0, stack, C);
  stack_push(1, stack, B);
  stack_push(2, stack, A);

  pthread_create(&threads[0], NULL, &thread0, NULL);
  pthread_create(&threads[1], NULL, &thread1, NULL);
  pthread_create(&threads[2], NULL, &thread2, NULL);

  for (int i = 0; i < NB_THREADS; i++)
    {
      pthread_join(threads[i], NULL);
    }
  if(stack->head != C){
    aba_detected = 1;
  }

  success = aba_detected;
  return success;
#else
  // No ABA is possible with lock-based synchronization. Let the test succeed only
  return 1;
#endif
}counter after push-3605 
#endif

// We test here the CAS function
struct thread_test_cas_args
{
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void*
thread_test_cas(void* arg)
{
#if NON_BLOCKING != 0
  thread_test_cas_args_t *args = (thread_test_cas_args_t*) arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++)
    {
      do {
        old = *args->counter;
        local = old + 1;
#if NON_BLOCKING == 1
      } while (cas(args->counter, old, local) != old);
#elif NON_BLOCKING == 2
      } while (software_cas(args->counter, old, local, args->lock) != old);
#endif
    }
#endif

  return NULL;
}

// Make sure Compare-and-swap works as expected
int
test_cas()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  size_t counter;

  int i, success;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (i = 0; i < NB_THREADS; i++)
    {
      args[i].id = i;
      args[i].counter = &counter;
      args[i].lock = &lock;
      pthread_create(&thread[i], &attr, &thread_test_cas, (void*) &args[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  success = assert(counter == (size_t)(NB_THREADS * MAX_PUSH_POP));

  if (!success)
    {
      printf("Got %ti, expected %i. ", counter, NB_THREADS * MAX_PUSH_POP);
    }

  return success;
#else
  return 1;
#endif
}

int
main(int argc, char **argv)
{
setbuf(stdout, NULL);
// MEASURE == 0 -> run unit tests
#if MEASURE == 0
  test_init();

  test_run(test_cas);

  test_run(test_push_safe);
  test_run(test_pop_safe);
  test_run(test_aba);
  test_finalize();
#else
  int i;
  pthread_t thread[NB_THREADS];
  pthread_attr_t attr;
  stack_measure_arg_t arg[NB_THREADS];
  pthread_attr_init(&attr);
  stack = stack_init();
  pool = stack_init();
  #if MEASURE == 1
  node_t *test;
  
  for (i = 0; i < MAX_PUSH_POP; i++){
    
      test = malloc(sizeof(test));
      stack_push(i, stack, test);
  }
  for (i = 0; i < 100; i++){
    
      test = malloc(sizeof(test));
      stack_push(i, pool, test);
  }
  for (i = 0; i < 100; i++){
      atomic_counter++;
      stack_pop(pool);
  }
  #endif

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < NB_THREADS; i++)
    {
      arg[i].id = i;
#if MEASURE == 1
      pthread_create(&thread[i], &attr, stack_measure_pop, (void*)&arg[i]);
#else
      pthread_create(&thread[i], &attr, stack_measure_push, (void*)&arg[i]);
#endif
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }
  clock_gettime(CLOCK_MONOTONIC, &stop);

  // Print out results
  for (i = 0; i < NB_THREADS; i++)
    {
        printf("%f\n", timediff(&t_start[i], &t_stop[i]));
    }
#endif

  return 0;
}
