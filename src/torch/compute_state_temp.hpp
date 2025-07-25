#pragma once
#ifndef COMPUTE_STATE_TEMP_HPP
#define COMPUTE_STATE_TEMP_CPP

namespace compute{

namespace state{

class Temp: public Base{
private:
public:
	//==== constructors/destructors ====
	Temp():Base(Name::TEMP){}
	~Temp(){}
	
	//==== operators ====
	friend std::ostream& operator<<(std::ostream& out, const Temp& temp);
	
	//==== member functions ====
	void read(Token& token){}
	double compute(const Struc& struc);
};

}

}

#endif