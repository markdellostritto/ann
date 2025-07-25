#pragma once
#ifndef OPENMP_HPP
#define OPENMP_HPP

#ifdef _OPENMP
	#include <omp.h>
#else
	#define omp_get_thread_num() 0
	#define omp_get_num_threads() 1
#endif

#endif