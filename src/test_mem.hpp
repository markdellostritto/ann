#ifndef TEST_MEM_HPP
#define TEST_MEM_HPP

template <class T>
void test_mem(const T& obj){
	std::cout<<"err - size - pack   = "<<serialize::error_size_pack(obj)<<"\n";
	std::cout<<"err - size - unpack = "<<serialize::error_size_unpack(obj)<<"\n";
	std::cout<<"err - arr           = "<<serialize::error_arr(obj)<<"\n";
}

#endif