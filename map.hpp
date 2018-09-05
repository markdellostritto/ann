#ifndef MAP_HPP
#define MAP_HPP

template <class T1, class T2>
class Map{
private:
	std::vector<T1> key_;
	std::vector<T2> val_;
public:
	//constructors/destructors
	Map(){};
	~Map(){};
	
	//access
	const std::vector<T1>& key()const{return key_;};
	const T1& key(unsigned int i)const{return key_[i];};
	const std::vector<T2>& val()const{return val_;};
	const T2& val(unsigned int i)const{return val_[i];};
	
	//member functions
	unsigned int size()const{return key_.size();};
	void add(const T1& k, const T2& v);
	void remove(const T1& k);
	bool find(const T1& k);
	void clear();
	
	//operators
	const T2& operator[](const T1& k)const;
};

template <class T1, class T2>
void Map<T1,T2>::add(const T1& k, const T2& v){
	key_.push_back(k);
	val_.push_back(v);
}

template <class T1, class T2>
void Map<T1,T2>::remove(const T1& k){
	int index=-1;
	for(unsigned int i=0; i<key_.size(); ++i){
		if(k==key_[i]){index=i;break;}
	}
	if(index>=0){
		key_.erase(key_.begin()+index);
		val_.erase(val_.begin()+index);
	}
}

template <class T1, class T2>
bool Map<T1,T2>::find(const T1& k){
	int index=-1;
	for(unsigned int i=0; i<key_.size(); ++i){
		if(k==key_[i]){index=i;break;}
	}
	if(index<0) return false;
	return true;
}

template <class T1, class T2>
void Map<T1,T2>::clear(){
	key_.clear();
	val_.clear();
}

template <class T1, class T2>
const T2& Map<T1,T2>::operator[](const T1& k)const{
	int index=-1;
	for(unsigned int i=0; i<key_.size(); ++i){
		if(k==key_[i]){index=i;break;}
	}
	if(index<0) throw std::runtime_error("No match for key in map");
	return val_[index];
}

#endif