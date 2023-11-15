/*
 * stack.c
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
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "non_blocking.h"
#include "stack.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif

stack_t *stack_alloc() {
  stack_t *stack = calloc(1, sizeof(stack_t));
#if NON_BLOCKING == 0
  pthread_mutex_init(&stack->stack_lock, NULL);
#endif
  pthread_mutex_init(&stack->free_lock, NULL);
  stack->blocks = calloc(100, sizeof(node_t*));
  stack->blocks_cap = 100;
  stack->blocks[0] = calloc(BLOCK_SIZE, sizeof(node_t));
  stack->blocks_size = 1;
  return stack;
}

int stack_check(stack_t *stack) {
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
  // This test fails if the task is not allocated or if the allocation failed
  assert(stack != NULL);

#if NON_BLOCKING != 0
  node_t *node = stack->head;
  while (node) {
    assert(node != node->next);
    node = node->next;
  }
#endif
#endif

  return 1;
}

node_t* stack_get_node(stack_t* stack) {
  pthread_mutex_lock(&stack->free_lock);
  size_t blockidx = stack->n / BLOCK_SIZE;
  if (blockidx > stack->blocks_size) {
    if (blockidx > stack->blocks_cap) {
      stack->blocks = realloc(stack->blocks, stack->blocks_cap * 2);
      stack->blocks_cap = stack->blocks_cap * 2;
    }
    stack->blocks[blockidx] = calloc(BLOCK_SIZE, sizeof(node_t));
    stack->blocks_size++;
  }
  size_t elemidx = stack->n % BLOCK_SIZE;
  node_t* node = &stack->blocks[blockidx][elemidx];
  stack->n++;
  pthread_mutex_unlock(&stack->free_lock);
  return node;
}

void stack_push(stack_t *stack, int val) {
#if NON_BLOCKING == 0
  node_t* node = stack_get_node(stack);
  node->val = val;

  pthread_mutex_lock(&stack->stack_lock);
  node->next = stack->head;
  stack->head = node;
  pthread_mutex_unlock(&stack->stack_lock);
#elif NON_BLOCKING == 1
  node_t* node = stack_get_node(stack);
  node->val = val;

  node_t* old;
  do {
    old = stack->head;
    node->next = old;
  } while (cas(&stack->head, old, node) != old);
#endif

  stack_check(stack);
}

int stack_pop(stack_t *stack) {
  int val;
#if NON_BLOCKING == 0
  pthread_mutex_lock(&stack->stack_lock);
  node_t *old_head = stack->head;

  if (old_head) {
    pthread_mutex_unlock(&stack->stack_lock);
    printf("pop from empty stack");
    return -1;
  }

  // Pop from stack
  stack->head = old_head->next;
  pthread_mutex_unlock(&stack->stack_lock);

  val = old_head->val;
  old_head->next = NULL;

  pthread_mutex_lock(&stack->free_lock);
  stack->n--;
  pthread_mutex_unlock(&stack->free_lock);
#elif NON_BLOCKING == 1
  node_t *old;
  node_t *new;
  do {
    old = stack->head;
    if (!old)
      printf("Head is NULL (will segfault)");
    new = old->next;
  } while (cas(&stack->head, old, new) != old);
  old->next = NULL;

  pthread_mutex_lock(&stack->free_lock);
  stack->n--;
  pthread_mutex_unlock(&stack->free_lock);
#endif

  stack_check(stack);

  return val;
}

void stack_print(stack_t* stack) {
  printf("Stack: ");
  node_t* node = stack->head;
  while (node) {
    printf("[%c, %p] -> %p ", node->val, node, node->next);
    node = node->next;
  }
  printf("\n");
}

void stack_free(stack_t *stack) {
  for (int i = 0; i < stack->blocks_size; ++i) {
    free(stack->blocks[i]);
  }
  free(stack->blocks);
  free(stack);
}
