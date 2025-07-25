// c++
#include <iostream>
#include <stdexcept>
// math
#include "math/eigen.hpp"
// str
#include "str/print.hpp"
// opt
#include "opt/iter.hpp"

namespace opt{

//***************************************************
//Iterator class
//***************************************************

std::ostream& operator<<(std::ostream& out, const Iterator& obj){
	char* str=new char[print::len_buf];
	out<<print::buf(str)<<"\n";
	out<<print::title("OPT::ITERATOR",str)<<"\n";
	//count
	out<<"N-PRINT = "<<obj.nPrint_<<"\n";
	out<<"N-WRITE = "<<obj.nWrite_<<"\n";
	out<<"STEP    = "<<obj.step_<<"\n";
	out<<"COUNT   = "<<obj.count_<<"\n";
	//stopping
	out<<"MAX     = "<<obj.max_<<"\n";
	out<<"STOP    = "<<obj.stop_<<"\n";
	out<<"LOSS    = "<<obj.loss_<<"\n";
	out<<"TOL     = "<<obj.tol_<<"\n";
	out<<print::buf(str);
	delete[] str;
	return out;
}

void Iterator::defaults(){
	if(OPT_ITER_PRINT_FUNC>0) std::cout<<"Iterator::defaults()\n";
	//count
		nPrint_=0;
		nWrite_=0;
		step_=0;
		count_=0;
	//stopping
		tol_=0;
		max_=0;
		stop_=Stop::UNKNOWN;
		loss_=Loss::UNKNOWN;
}

}

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const opt::Iterator& obj){
	if(OPT_ITER_PRINT_FUNC>0) std::cout<<"serialize::nbytes(const opt::Iterator&)\n";
	int size=0;
	//count
		size+=sizeof(int);//nPrint_
		size+=sizeof(int);//nWrite_
		size+=sizeof(int);//step_
		size+=sizeof(int);//count_
	//stopping
		size+=sizeof(int);//max_
		size+=sizeof(opt::Stop);//stop_
		size+=sizeof(opt::Loss);//loss_
		size+=sizeof(double);//tol_
	//return the size
	return size;
}

//**********************************************
// packing
//**********************************************

template <> int pack(const opt::Iterator& obj, char* arr){
	if(OPT_ITER_PRINT_FUNC>0) std::cout<<"serialize::pack(const opt::Iterator&,char*)\n";
	int pos=0;
	//count
		std::memcpy(arr+pos,&obj.nPrint(),sizeof(int)); pos+=sizeof(int);//nPrint_
		std::memcpy(arr+pos,&obj.nWrite(),sizeof(int)); pos+=sizeof(int);//nWrite_
		std::memcpy(arr+pos,&obj.step(),sizeof(int)); pos+=sizeof(int);//step_
		std::memcpy(arr+pos,&obj.count(),sizeof(int)); pos+=sizeof(int);//count_
	//stopping
		std::memcpy(arr+pos,&obj.max(),sizeof(int)); pos+=sizeof(int);//max_
		std::memcpy(arr+pos,&obj.stop(),sizeof(opt::Stop)); pos+=sizeof(opt::Stop);//stop_
		std::memcpy(arr+pos,&obj.loss(),sizeof(opt::Loss)); pos+=sizeof(opt::Loss);//loss_
		std::memcpy(arr+pos,&obj.tol(),sizeof(double)); pos+=sizeof(double);//tol_
	//return bytes written
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(opt::Iterator& obj, const char* arr){
	if(OPT_ITER_PRINT_FUNC>0) std::cout<<"serialize::unpack(opt::Iterator&,const char*)\n";
	int pos=0;
	//count
		std::memcpy(&obj.nPrint(),arr+pos,sizeof(int)); pos+=sizeof(int);//nPrint_
		std::memcpy(&obj.nWrite(),arr+pos,sizeof(int)); pos+=sizeof(int);//nWrite_
		std::memcpy(&obj.step(),arr+pos,sizeof(int)); pos+=sizeof(int);//step_
		std::memcpy(&obj.count(),arr+pos,sizeof(int)); pos+=sizeof(int);//count_
	//stopping
		std::memcpy(&obj.max(),arr+pos,sizeof(int)); pos+=sizeof(int);//max_
		std::memcpy(&obj.stop(),arr+pos,sizeof(opt::Stop)); pos+=sizeof(opt::Stop);//stop_
		std::memcpy(&obj.loss(),arr+pos,sizeof(opt::Loss)); pos+=sizeof(opt::Loss);//loss_
		std::memcpy(&obj.tol(),arr+pos,sizeof(double)); pos+=sizeof(double);//tol_
	//return bytes read
	return pos;
}

}