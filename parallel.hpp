#pragma once
#ifndef PARALLEL_HPP
#define PARALLEL_HPP

#include <vector>

namespace parallel{

std::vector<unsigned int>& gen_thread_dist(unsigned int nThreads, unsigned int nObj, std::vector<unsigned int>& thread_dist);
std::vector<unsigned int>& gen_thread_offset(unsigned int nThreads, unsigned int nObj, std::vector<unsigned int>& thread_offset);
int* gen_thread_dist(unsigned int nThreads, unsigned int nObj, int* thread_dist);
int* gen_thread_offset(unsigned int nThreads, unsigned int nObj, int* thread_offset);
unsigned int thread_subset(unsigned int N, unsigned int rank, unsigned int nproc);
unsigned int thread_offset(unsigned int N, unsigned int rank, unsigned int nproc);
}

#endif