// c libraries
#include <cstring>
// ann - serialize
#include "serialize.h"

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> int nbytes(const bool& obj){return sizeof(obj);}
template <> int nbytes(const char& obj){return sizeof(obj);}
template <> int nbytes(const unsigned char& obj){return sizeof(obj);}
template <> int nbytes(const short& obj){return sizeof(obj);}
template <> int nbytes(const unsigned short& obj){return sizeof(obj);}
template <> int nbytes(const int& obj){return sizeof(obj);}
template <> int nbytes(const unsigned int& obj){return sizeof(obj);}
template <> int nbytes(const float& obj){return sizeof(obj);}
template <> int nbytes(const double& obj){return sizeof(obj);}
template <> int nbytes(const std::string& str){return sizeof(char)*str.length()+sizeof(int);}//size + length
template <> int nbytes(const std::vector<std::string>& strlist){
	int size=0;
	size+=sizeof(int);
	for(int i=0; i<strlist.size(); ++i){
		size+=sizeof(int);
		size+=strlist[i].length();
	}
	return size;
}
template <> int nbytes(const std::vector<int>& vec){
	int size=0;
	size+=sizeof(int);//size
	size+=sizeof(int)*vec.size();
	return size;
}
template <> int nbytes(const std::vector<unsigned int>& vec){
	int size=0;
	size+=sizeof(int);//size
	size+=sizeof(unsigned int)*vec.size();
	return size;
}
template <> int nbytes(const std::vector<double>& vec){
	int size=0;
	size+=sizeof(int);//size
	size+=sizeof(double)*vec.size();
	return size;
}
	
//**********************************************
// packing
//**********************************************

template <> int pack(const bool& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(bool);}
template <> int pack(const char& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(char);}
template <> int pack(const unsigned char& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(unsigned char);}
template <> int pack(const short& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(short);}
template <> int pack(const unsigned short& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(unsigned short);}
template <> int pack(const int& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(int);}
template <> int pack(const unsigned int& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(unsigned int);}
template <> int pack(const float& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(float);}
template <> int pack(const double& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(double);}
template <> int pack(const std::string& str, char* arr){
	int pos=0;
	int length=str.length();
	std::memcpy(arr+pos,&length,sizeof(int)); pos+=sizeof(int);
	std::memcpy(arr+pos,str.c_str(),sizeof(char)*str.length()); pos+=sizeof(char)*str.length();
	return pos;
}
template <> int pack(const std::vector<std::string>& strlist, char* arr){
	int pos=0;
	int size;
	std::memcpy(arr+pos,&(size=strlist.size()),sizeof(int)); pos+=sizeof(int);
	for(int i=0; i<strlist.size(); ++i){
		std::memcpy(arr+pos,&(size=strlist[i].length()),sizeof(int)); pos+=sizeof(int);
		std::memcpy(arr+pos,strlist[i].c_str(),sizeof(char)*strlist[i].length()); pos+=sizeof(char)*strlist[i].length();
	}
	return pos;
}
template <> int pack(const std::vector<int>& vec, char* arr){
	int pos=0;
	int temp=vec.size();
	std::memcpy(arr+pos,&temp,sizeof(int)); pos+=sizeof(int);
	for(int i=0; i<vec.size(); ++i){
		std::memcpy(arr+pos,&vec[i],sizeof(int)); pos+=sizeof(int);
	}
	return pos;
}
template <> int pack(const std::vector<unsigned int>& vec, char* arr){
	int pos=0;
	int temp=vec.size();
	std::memcpy(arr+pos,&temp,sizeof(int)); pos+=sizeof(int);
	for(int i=0; i<vec.size(); ++i){
		std::memcpy(arr+pos,&vec[i],sizeof(unsigned int)); pos+=sizeof(unsigned int);
	}
	return pos;
}
template <> int pack(const std::vector<double>& vec, char* arr){
	int pos=0;
	int temp=vec.size();
	std::memcpy(arr+pos,&temp,sizeof(int)); pos+=sizeof(int);
	for(int i=0; i<vec.size(); ++i){
		std::memcpy(arr+pos,&vec[i],sizeof(double)); pos+=sizeof(double);
	}
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> int unpack(bool& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(bool);}
template <> int unpack(char& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(char);}
template <> int unpack(unsigned char& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(unsigned char);}
template <> int unpack(short& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(short);}
template <> int unpack(unsigned short& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(unsigned short);}
template <> int unpack(int& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(int);}
template <> int unpack(unsigned int& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(unsigned int);}
template <> int unpack(float& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(float);}
template <> int unpack(double& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(double);}
template <> int unpack(std::string& str, const char* arr){
	int pos=0;
	int length=0;
	std::memcpy(&length,arr+pos,sizeof(int)); pos+=sizeof(int);
	if(length>0){
		str.resize(length);
		for(int i=0; i<length; ++i){
			std::memcpy(&(str[i]),arr+pos,sizeof(char)); pos+=sizeof(char);
		}
	} else str.clear();
	return pos;
}
template <> int unpack(std::vector<std::string>& strlist, const char* arr){
	int pos=0;
	int size=0;
	std::memcpy(&size,arr+pos,sizeof(int)); pos+=sizeof(int);
	strlist.resize(size);
	if(size>0){
		for(int i=0; i<strlist.size(); ++i){
			std::memcpy(&size,arr+pos,sizeof(int)); pos+=sizeof(int);
			strlist[i].resize(size);
			for(int j=0; j<strlist[i].length(); ++j){
				std::memcpy(&(strlist[i][j]),arr+pos,sizeof(char)); pos+=sizeof(char);
			}
		}
	} else strlist.clear();
	return pos;
}
template <> int unpack(std::vector<int>& vec, const char* arr){
	int pos=0;
	int size=0;
	std::memcpy(&size,arr+pos,sizeof(int)); pos+=sizeof(int);
	if(size>0){
		vec.resize(size);
		for(int i=0; i<vec.size(); ++i){
			std::memcpy(&vec[i],arr+pos,sizeof(int)); pos+=sizeof(int);
		}
	} else vec.clear();
	return pos;
}
template <> int unpack(std::vector<unsigned int>& vec, const char* arr){
	int pos=0;
	int size=0;
	std::memcpy(&size,arr+pos,sizeof(int)); pos+=sizeof(int);
	if(size>0){
		vec.resize(size);
		for(int i=0; i<vec.size(); ++i){
			std::memcpy(&vec[i],arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		}
	} else vec.clear();
	return pos;
}
template <> int unpack(std::vector<double>& vec, const char* arr){
	int pos=0;
	int size=0;
	std::memcpy(&size,arr+pos,sizeof(int)); pos+=sizeof(int);
	if(size>0){
		vec.resize(size);
		for(int i=0; i<vec.size(); ++i){
			std::memcpy(&vec[i],arr+pos,sizeof(double)); pos+=sizeof(double);
		}
	} else vec.clear();
	return pos;
}

}