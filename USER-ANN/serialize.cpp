#include "serialize.h"

namespace serialize{

//**********************************************
// byte measures
//**********************************************

template <> unsigned int nbytes(const bool& obj){return sizeof(obj);};
template <> unsigned int nbytes(const char& obj){return sizeof(obj);};
template <> unsigned int nbytes(const unsigned char& obj){return sizeof(obj);};
template <> unsigned int nbytes(const short& obj){return sizeof(obj);};
template <> unsigned int nbytes(const unsigned short& obj){return sizeof(obj);};
template <> unsigned int nbytes(const int& obj){return sizeof(obj);};
template <> unsigned int nbytes(const unsigned int& obj){return sizeof(obj);};
template <> unsigned int nbytes(const float& obj){return sizeof(obj);};
template <> unsigned int nbytes(const double& obj){return sizeof(obj);};
template <> unsigned int nbytes(const std::string& str){return sizeof(char)*str.length()+sizeof(unsigned int);}//size + length
template <> unsigned int nbytes(const std::vector<std::string>& strlist){
	unsigned int size=0;
	size+=sizeof(unsigned int);
	for(unsigned int i=0; i<strlist.size(); ++i){
		size+=sizeof(unsigned int);
		size+=strlist[i].length();
	}
	return size;
}
template <> unsigned int nbytes(const std::vector<unsigned int>& vec){
	unsigned int size=0;
	size+=sizeof(unsigned int);//size
	size+=sizeof(unsigned int)*vec.size();
	return size;
}
template <> unsigned int nbytes(const std::vector<double>& vec){
	unsigned int size=0;
	size+=sizeof(unsigned int);//size
	size+=sizeof(double)*vec.size();
	return size;
}
	
//**********************************************
// packing
//**********************************************

template <> unsigned int pack(const bool& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(bool);};
template <> unsigned int pack(const char& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(char);};
template <> unsigned int pack(const unsigned char& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(unsigned char);};
template <> unsigned int pack(const short& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(short);};
template <> unsigned int pack(const unsigned short& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(unsigned short);};
template <> unsigned int pack(const int& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(int);};
template <> unsigned int pack(const unsigned int& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(unsigned int);};
template <> unsigned int pack(const float& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(float);};
template <> unsigned int pack(const double& obj, char* arr){std::memcpy(arr,&obj,nbytes(obj)); return sizeof(double);};
template <> unsigned int pack(const std::string& str, char* arr){
	unsigned int pos=0;
	unsigned int length=str.length();
	std::memcpy(arr+pos,&length,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	std::memcpy(arr+pos,str.c_str(),sizeof(char)*str.length()); pos+=sizeof(char)*str.length();
	return pos;
}
template <> unsigned int pack(const std::vector<std::string>& strlist, char* arr){
	unsigned int pos=0;
	unsigned int size;
	std::memcpy(arr+pos,&(size=strlist.size()),sizeof(unsigned int)); pos+=sizeof(unsigned int);
	for(unsigned int i=0; i<strlist.size(); ++i){
		std::memcpy(arr+pos,&(size=strlist[i].length()),sizeof(unsigned int)); pos+=sizeof(unsigned int);
		std::memcpy(arr+pos,strlist[i].c_str(),sizeof(char)*strlist[i].length()); pos+=sizeof(char)*strlist[i].length();
	}
	return pos;
}
template <> unsigned int pack(const std::vector<unsigned int>& vec, char* arr){
	unsigned int pos=0;
	unsigned int temp=vec.size();
	std::memcpy(arr+pos,&temp,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	for(unsigned int i=0; i<vec.size(); ++i){
		std::memcpy(arr+pos,&vec[i],sizeof(unsigned int)); pos+=sizeof(unsigned int);
	}
	return pos;
}
template <> unsigned int pack(const std::vector<double>& vec, char* arr){
	unsigned int pos=0;
	unsigned int temp=vec.size();
	std::memcpy(arr+pos,&temp,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	for(unsigned int i=0; i<vec.size(); ++i){
		std::memcpy(arr+pos,&vec[i],sizeof(double)); pos+=sizeof(double);
	}
	return pos;
}

//**********************************************
// unpacking
//**********************************************

template <> unsigned int unpack(bool& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(bool);};
template <> unsigned int unpack(char& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(char);};
template <> unsigned int unpack(unsigned char& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(unsigned char);};
template <> unsigned int unpack(short& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(short);};
template <> unsigned int unpack(unsigned short& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(unsigned short);};
template <> unsigned int unpack(int& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(int);};
template <> unsigned int unpack(unsigned int& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(unsigned int);};
template <> unsigned int unpack(float& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(float);};
template <> unsigned int unpack(double& obj, const char* arr){std::memcpy(&obj,arr,nbytes(obj)); return sizeof(double);};
template <> unsigned int unpack(std::string& str, const char* arr){
	unsigned int pos=0;
	unsigned int length=0;
	std::memcpy(&length,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	if(length>0){
		str.resize(length);
		for(unsigned int i=0; i<length; ++i){
			std::memcpy(&(str[i]),arr+pos,sizeof(char)); pos+=sizeof(char);
		}
	} else str.clear();
	return pos;
}
template <> unsigned int unpack(std::vector<std::string>& strlist, const char* arr){
	unsigned int pos=0;
	unsigned int size=0;
	std::memcpy(&size,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	strlist.resize(size);
	if(size>0){
		for(unsigned int i=0; i<strlist.size(); ++i){
			std::memcpy(&size,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
			strlist[i].resize(size);
			for(unsigned int j=0; j<strlist[i].length(); ++j){
				std::memcpy(&(strlist[i][j]),arr+pos,sizeof(char)); pos+=sizeof(char);
			}
		}
	} else strlist.clear();
	return pos;
}
template <> unsigned int unpack(std::vector<unsigned int>& vec, const char* arr){
	unsigned int pos=0;
	unsigned int size=0;
	std::memcpy(&size,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	if(size>0){
		vec.resize(size);
		for(unsigned int i=0; i<vec.size(); ++i){
			std::memcpy(&vec[i],arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
		}
	} else vec.clear();
	return pos;
}
template <> unsigned int unpack(std::vector<double>& vec, const char* arr){
	unsigned int pos=0;
	unsigned int size=0;
	std::memcpy(&size,arr+pos,sizeof(unsigned int)); pos+=sizeof(unsigned int);
	if(size>0){
		vec.resize(size);
		for(unsigned int i=0; i<vec.size(); ++i){
			std::memcpy(&vec[i],arr+pos,sizeof(double)); pos+=sizeof(unsigned);
		}
	} else vec.clear();
	return pos;
}

}