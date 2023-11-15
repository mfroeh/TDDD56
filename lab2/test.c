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

#include <assert.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "non_blocking.h"
#include "stack.h"
#include "test.h"

#define test_run(test)                                                         \
  printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, \
         #test);                                                               \
  test_setup();                                                                \
  if (test()) {                                                                \
    printf("passed\n");                                                        \
  } else {                                                                     \
    printf("failed\n");                                                        \
  }                                                                            \
  test_teardown();

/* Helper function for measurement */
double timediff(struct timespec *begin, struct timespec *end) {
  double sec = 0.0, nsec = 0.0;
  if ((end->tv_nsec - begin->tv_nsec) < 0) {
    sec = (double)(end->tv_sec - begin->tv_sec - 1);
    nsec = (double)(end->tv_nsec - begin->tv_nsec + 1000000000);
  } else {
    sec = (double)(end->tv_sec - begin->tv_sec);
    nsec = (double)(end->tv_nsec - begin->tv_nsec);
  }
  return sec + nsec / 1E9;
}

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

#ifndef NDEBUG
int assert_fun(int expr, const char *str, const char *file,
               const char *function, size_t line) {
  if (!(expr)) {
    fprintf(stderr, "[%s:%s:%zu][ERROR] Assertion failure: %s\n", file,
            function, line, str);
    abort();
    // If some hack disables abort above
    return 0;
  } else
    return 1;
}
#endif

stack_t *stack;
data_t data;

#if MEASURE != 0
struct stack_measure_arg {
  int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;

#if MEASURE == 1
void *stack_measure_pop(void *arg) {
  stack_measure_arg_t *args = (stack_measure_arg_t *)arg;
  int i;

  clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
  for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++) {
    // See how fast your implementation can pop MAX_PUSH_POP elements in
    // parallel
    stack_pop(stack);
  }
  clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

  return NULL;
}
#elif MEASURE == 2
void *stack_measure_push(void *arg) {
  stack_measure_arg_t *args = (stack_measure_arg_t *)arg;
  int i;

  clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
  for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++) {
    // See how fast your implementation can push MAX_PUSH_POP elements in
    // parallel
    stack_push(stack, i);
  }
  clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

  return NULL;
}
#endif
#endif

/* A bunch of optional (but useful if implemented) unit tests for your stack */
void test_init() {
  // Initialize your test batch
}

void test_setup() {
  data = DATA_VALUE;

  stack = stack_alloc();
}

void test_teardown() {
  if (!stack_check(stack)) {
    printf("Stack corrupt!");
  } else {
    stack_free(stack);
  }
}

void test_finalize() {
  // Destroy properly your test batch
}

int test_push_safe() {
  // Make sure your stack remains in a good state with expected content when
  // several threads push concurrently to it

  stack_push(stack, 31);

  // check if the stack is in a consistent state
  int res = assert(stack_check(stack));

  // check other properties expected after a push operation
  // (this is to be updated as your stack design progresses)
  // Now, the test succeeds
  return res && assert(stack->head->val == 31);
}

int test_pop_safe() {
  // Same as the test above for parallel pop operation
  int res = assert(stack_check(stack));

  // For now, this test always fails
  return res;
}

// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 3

pthread_barrier_t aba_barrier;

#if NON_BLOCKING != 0
void thread0() {
  node_t *A = stack->head;
  node_t *B = A->next;
  pthread_barrier_wait(&aba_barrier); // Simulate interrupt
  pthread_barrier_wait(&aba_barrier); // Simulate thread2 executes after thread1
  pthread_barrier_wait(
      &aba_barrier); // Simulate both thread1 and thread2 popped
  pthread_barrier_wait(&aba_barrier); // Simulate thread1 reuses 'A' element
  printf("Before thread 0 CAS\n");
  stack_print(stack);
  assert(cas(&stack->head, A, B) == A); // ABA: We successfully pop A
  printf("After thread 0 CAS\n");
  stack_print(stack);
  stack->n--;
  assert(stack->head == B);             // ABA: Stacks head points to B
  assert(&stack->blocks[stack->n / BLOCK_SIZE][stack->n % BLOCK_SIZE] == B) // First element in the freelist is B
  // Both stack and freelist "point" to B!
  return 1;
}

void thread1() {
  pthread_barrier_wait(
      &aba_barrier); // Simulate begin execution after thread0 interrupt
  stack_pop(stack);  // Pops A
  pthread_barrier_wait(&aba_barrier); // Simulate thread2 executes after thread1
  pthread_barrier_wait(
      &aba_barrier);      // Simulate both thread1 and thread2 popped
  stack->aba++;
  printf("Before thread 1 reuse A\n");
  stack_print(stack);
  stack_push(stack, 'D'); // Push reused 'A' element
  printf("After thread 1 reuse A\n");
  stack_print(stack);
  stack->aba--;
  pthread_barrier_wait(&aba_barrier); // Simulate thread1 reuses 'A' element
}

void thread2() {
  pthread_barrier_wait(
      &aba_barrier); // Simulate begin execution after thread0 interrupt
  pthread_barrier_wait(&aba_barrier); // Simulate thread2 executes after thread1
  stack_pop(stack);                   // Pops B
  pthread_barrier_wait(
      &aba_barrier); // Simulate both thread1 and thread 2 popped
  pthread_barrier_wait(&aba_barrier); // Simulate thread1 reuses 'A' element
}
#endif

int test_aba() {
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  stack = stack_alloc();
  stack_push(stack, 'C');
  stack_push(stack, 'B');
  stack_push(stack, 'A');

  printf("Before test start\n");
  stack_print(stack);
  assert(pthread_barrier_init(&aba_barrier, NULL, 3) == 0);

  pthread_attr_t attr;
  pthread_attr_init(&attr);

  pthread_t thread_0;
  pthread_create(&thread_0, &attr, thread0, NULL);

  pthread_t thread_1;
  pthread_create(&thread_1, &attr, thread1, NULL);

  pthread_t thread_2;
  pthread_create(&thread_2, &attr, thread2, NULL);

  pthread_join(thread_0, NULL);
  pthread_join(thread_1, NULL);
  pthread_join(thread_2, NULL);

  // We assert that ABA occurs inside of thread0 func
  return 1;
#else
  // No ABA is possible with lock-based synchronization. Let the test succeed
  // only
  return 1;
#endif
}

// We test here the CAS function
struct thread_test_cas_args {
  int id;
  size_t *counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void *thread_test_cas(void *arg) {
#if NON_BLOCKING != 0
  thread_test_cas_args_t *args = (thread_test_cas_args_t *)arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++) {
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
int test_cas() {
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

  for (i = 0; i < NB_THREADS; i++) {
    args[i].id = i;
    args[i].counter = &counter;
    args[i].lock = &lock;
    pthread_create(&thread[i], &attr, &thread_test_cas, (void *)&args[i]);
  }

  for (i = 0; i < NB_THREADS; i++) {
    pthread_join(thread[i], NULL);
  }

  success = assert(counter == (size_t)(NB_THREADS * MAX_PUSH_POP));

  if (!success) {
    printf("Got %ti, expected %i. ", counter, NB_THREADS * MAX_PUSH_POP);
  }

  return success;
#else
  return 1;
#endif
}

int main(int argc, char **argv) {
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

  stack = stack_alloc();
#if MEASURE == 1
  for (int i = 0; i < MAX_PUSH_POP; ++i) {
    stack_push(stack, i);
  }
#endif

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < NB_THREADS; i++) {
    arg[i].id = i;
#if MEASURE == 1
    pthread_create(&thread[i], &attr, stack_measure_pop, (void *)&arg[i]);
#else
    pthread_create(&thread[i], &attr, stack_measure_push, (void *)&arg[i]);
#endif
  }

  for (i = 0; i < NB_THREADS; i++) {
    pthread_join(thread[i], NULL);
  }
  clock_gettime(CLOCK_MONOTONIC, &stop);

  // Print out results
  for (i = 0; i < NB_THREADS; i++) {
    printf("Thread %d time: %f\n", i, timediff(&t_start[i], &t_stop[i]));
  }
#endif

  return 0;
}
