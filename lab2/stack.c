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
  stack_t *stack = malloc(sizeof(stack_t));
#if NON_BLOCKING == 0
  pthread_mutex_init(&stack->lock, NULL);
#endif
  stack->head = NULL;
  stack->free_front = NULL;
  return stack;
}

int stack_check(stack_t *stack) {
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
  // This test fails if the task is not allocated or if the allocation failed
  assert(stack != NULL);

#if NON_BLOCKING != 0
  // (Yes this is N^2 sorry)
  node_t *node = stack->head;
  while (node) {
    node_t *free_node = stack->free_front;
    while (free_node) {
      if (node == free_node) return 0;
      free_node = free_node->next;
    }
    node = node->next;
  }
#endif
#endif

  return 1;
}

void stack_push(stack_t *stack, int val) {
#if NON_BLOCKING == 0
  pthread_mutex_lock(&stack->lock);

  node_t *node = stack->free_front;
  if (node) {
    stack->free_front = node->next;
  } else {
    node = malloc(sizeof(node_t));
  }

  node->val = val;
  node->next = stack->head;
  stack->head = node;

  pthread_mutex_unlock(&stack->lock);
#elif NON_BLOCKING == 1
  node_t *old;
  node_t *new;
  do {
    old = stack->free_front;
    new = old ? old->next : NULL;
  } while (cas(&stack->free_front, old, new) != old);

  node_t* node = old;
  if (!node)
    node = malloc(sizeof(node_t));
  node->val = val;

  do {
    old = stack->head;
    node->next = old;
  } while (cas(&stack->head, old, node) != old);

#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a
  // consistent check It doesn't harm performance as sanity check are disabled
  // at measurement time This is to be updated as your implementation progresses
  stack_check(stack);
}

int stack_pop(stack_t *stack) {
  int val;
#if NON_BLOCKING == 0
  pthread_mutex_lock(&stack->lock);

  node_t *old_head = stack->head;
  if (!old_head)
    printf("Head is NULL (will segfault)");

  // Pop from stack
  val = old_head->val;
  stack->head = old_head->next;

  // Append to freelist
  old_head->next = NULL;

  node_t* free_last = stack->free_front;
  if (!free_last) {
    stack->free_front = old_head;
  } else {
    while (free_last->next) {
      free_last = free_last->next;
    }
    free_last->next = old_head;
  }

  pthread_mutex_unlock(&stack->lock);
#elif NON_BLOCKING == 1
  node_t *old;
  node_t *new;
  do {
    old = stack->head;
    if (!old)
      printf("Head is NULL (will segfault)");
    new = old->next;
  } while (cas(&stack->head, old, new) != old);

  node_t* node = old;
  node->next = NULL;
  val = node->val;

  void** ptr = NULL;
  // Append to freelist
  do {
      node_t* free_last = stack->free_front;
      if (!free_last) {
        ptr = &stack->free_front;
      } else {
        while (free_last && free_last->next) {
          free_last = free_last->next;
        }
        ptr = &free_last->next;
      }

      old = free_last ? free_last->next : NULL;
      new = node;
  } while (cas(ptr, old, new) != old);
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  stack_check(stack);

  return val;
}

void stack_free(stack_t *stack) {
#if NON_BLOCKING == 0
  pthread_mutex_lock(&stack->lock);
#endif

  // Free the stack
  node_t *node = stack->head;
  while (node != NULL) {
    node_t* tmp = node;
    node = tmp->next;
    free(tmp);
  }

  // Free the freelist
  node = stack->free_front;
  while (node != NULL) {
    node_t* tmp = node;
    node = tmp->next;
    free(tmp);
  }

#if NON_BLOCKING == 0
  pthread_mutex_unlock(&stack->lock);
#endif

  free(stack);
}
