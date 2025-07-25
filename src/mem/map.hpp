#pragma once
#ifndef MAP_HPP
#define MAP_HPP

//c++ libraries
#include <vector>
#include <iosfwd>
//ann - serialize
#include "mem/serialize.hpp"

/**
* maps two different types of objects to each other
*/
template <class T1, class T2>
class Map{
private:
	std::vector<T1> key_;
	std::vector<T2> val_;
public:
	//constructors/destructors
	Map(){}
	~Map(){}
	
	//access
	const std::vector<T1>& key()const{return key_;}
	const T1& key(int i)const{return key_[i];}
	const std::vector<T2>& val()const{return val_;}
	const T2& val(int i)const{return val_[i];}
	
	//member functions
	int size()const{return key_.size();}
	void add(const T1& k, const T2& v);
	void remove(const T1& k);
	bool find(const T1& k)const;
	int index(const T1& k)const;
	void clear();
	
	//operators
	const T2& operator[](const T1& k)const;
};

/**
* add a key-value pair to the map
* @param k - key
* @param v - value
*/
template <class T1, class T2>
void Map<T1,T2>::add(const T1& k, const T2& v){
	key_.push_back(k);
	val_.push_back(v);
}

/**
* remove a key-value pair to the map
* @param k - key
* @param v - value
*/
template <class T1, class T2>
void Map<T1,T2>::remove(const T1& k){
	int index=-1;
	for(int i=0; i<key_.size(); ++i){
		if(k==key_[i]){index=i;break;}
	}
	if(index>=0){
		key_.erase(key_.begin()+index);
		val_.erase(val_.begin()+index);
	}
}

/**
* find a key in the map
* @param k - key
* @return whether the key is in the map
*/
template <class T1, class T2>
bool Map<T1,T2>::find(const T1& k)const{
	int index=-1;
	for(int i=0; i<key_.size(); ++i){
		if(k==key_[i]){index=i;break;}
	}
	if(index<0) return false;
	return true;
}

/**
* find a key in the map
* @param k - key
* @return the index of the key in the map
*/
template <class T1, class T2>
int Map<T1,T2>::index(const T1& k)const{
	int index_=-1;
	for(int i=0; i<key_.size(); ++i){
		if(k==key_[i]){index_=i;break;}
	}
	return index_;
}

/**
* clear all values in the map
*/
template <class T1, class T2>
void Map<T1,T2>::clear(){
	key_.clear();
	val_.clear();
}

/**
* return the value associated with a given key
* @param k - the key
* @param the value associated with the key, if the key is present
* @return the object associated with the key
* if there is no match for the key, an exception is thrown
*/
template <class T1, class T2>
const T2& Map<T1,T2>::operator[](const T1& k)const{
	int index=-1;
	for(int i=0; i<key_.size(); ++i){
		if(k==key_[i]){index=i;break;}
	}
	if(index<0) throw std::runtime_error("No match for key in map");
	return val_[index];
}

template <class T1, class T2>
bool operator==(const Map<T1,T2>& map1, const Map<T1,T2>& map2){
	if(map1.size()!=map2.size()) return false;
	for(int i=0; i<map1.size(); ++i){
		if(map1.val(i)!=map2[map1.key(i)]) return false;
	}
	return true;
}

template <class T1, class T2>
bool operator!=(const Map<T1,T2>& map1, const Map<T1,T2>& map2){
	return !(map1==map2);
}

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const Map<std::string,int>& obj);
	template <> int nbytes(const Map<int,int>& obj);
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const Map<std::string,int>& obj, char* arr);
	template <> int pack(const Map<int,int>& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(Map<std::string,int>& obj, const char* arr);
	template <> int unpack(Map<int,int>& obj, const char* arr);
	
}

#endif