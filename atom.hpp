#ifndef ATOM_HPP
#define ATOM_HPP

#ifndef __cplusplus
	#error A C++ compiler is required.
#endif

template <typename... Args>
class Atom: public Args...{
public:
	//constructors/destructors
	Atom():Args()...{};
	Atom(const Atom<Args...>& a):Args(a)...{};
	~Atom(){};
	//operators
	Atom<Args...>& operator=(const Atom<Args...>& a);
	//members
	void clear();
	void init();
};

template <typename... Args>
Atom<Args...>& Atom<Args...>::operator=(const Atom<Args...>& a){
	int arr[sizeof...(Args)]={(Args::operator=(a),0)...};
}

template <typename... Args>
void Atom<Args...>::clear(){
	int arr[sizeof...(Args)]={(Args::clear(),0)...};
}

template <typename... Args>
void Atom<Args...>::init(){
	int arr[sizeof...(Args)]={(Args::init(),0)...};
}

#endif