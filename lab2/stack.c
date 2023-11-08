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
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif

stack_t* stack_alloc(){
  stack_t* stack = malloc(sizeof(stack_t));
  pthread_mutex_init(&stack->lock, NULL);
  stack->head = NULL;
  stack->size = 2;
  return stack;
}

int
stack_check(stack_t *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
  // This test fails if the task is not allocated or if the allocation failed
	assert(stack != NULL);

#if NON_BLOCKING == 0
  return 1;
#else
  return 0;
#endif
#else
  return 1;
#endif
}

void /* Return the type you prefer */
stack_push(stack_t* stack, int val)
{
#if NON_BLOCKING == 0
  pthread_mutex_lock(&stack->lock);
  // Implement a lock_based stack
  node_t* node = calloc(1, sizeof(node_t));
  node->val = val;
  node->next = stack->head;
  stack->head = node;
  pthread_mutex_unlock(&stack->lock);
#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check((stack_t*)1);
}

int /* Return the type you prefer */
stack_pop(stack_t* stack)
{
#if NON_BLOCKING == 0
  pthread_mutex_lock(&stack->lock);
  // Implement a lock_based stack
  node_t* old_head = stack->head;
  // if (old_head == NULL)
  // {
  //   pthread_mutex_unlock(&stack->lock);
  //   // Special Case watch out
  //   return -1;
  // }
  
  int val = old_head->val;
  stack->head = old_head->next;
  free(old_head);
  pthread_mutex_unlock(&stack->lock);
#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  return val;
}

void stack_free(stack_t* stack){
  pthread_mutex_lock(&stack->lock);
  node_t* node = stack->head;
  while(node != NULL) {
    free(node);
    node = node->next;
  }
  pthread_mutex_unlock(&stack->lock);

  free(stack);
}

