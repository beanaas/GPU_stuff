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


int
stack_check(stack_t *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
	// Use assert() to check if your stack is in a state that makes sens
	// This test should always pass 
	assert(1 == 1);

	// This test fails if the task is not allocated or if the allocation failed
	assert(stack != NULL);
#endif
	// The stack is always fine
	return 1;
}


int /* Return the type you prefer */
stack_push(int val, stack_t *stack, node_t *node)
{
  node->val = val;
  node_t *old_head;

#if NON_BLOCKING == 0
  // Implement a lock_based stack
  pthread_mutex_lock(&stack->lock);
  old_head = stack->head;
  node->next = old_head;
  stack->head = node;
  pthread_mutex_unlock(&stack->lock);
  
#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  do{
    old_head = stack->head;
    node->next = old_head;
  }while((node_t*)cas((size_t*)&stack->head, (size_t)old_head, (size_t)node)!=old_head);
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  return 0;
}


node_t* stack_pop(stack_t *stack)
{
  node_t *node_to_pop;
#if NON_BLOCKING == 0
  pthread_mutex_lock(&stack->lock);
  node_to_pop=stack->head;
  stack->head = node_to_pop->next;
  pthread_mutex_unlock(&stack->lock);
  // Implement a lock_based stack
#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
	do {
		node_to_pop = stack->head;
	}	while(node_to_pop != (node_t*)cas((size_t*)&stack->head, (size_t)node_to_pop, (size_t)node_to_pop->next));
  
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif
  //res = temp.val; 
   stack_check((stack_t*)1);
  return node_to_pop;
}


stack_t* stack_init() 
{
  stack_t *stack = malloc(sizeof(stack_t));
  stack->head = NULL;
  pthread_mutex_init(&(stack->lock), NULL);

  return stack; 
}
