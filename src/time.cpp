//c++ libraries
#include <iostream>
#include "time.hpp"

//==== operator ====

std::ostream& operator<<(std::ostream& out, const Clock& c){
	return out<<"clock start "<<c.start_<<" stop "<<c.stop_<<" time "<<c.time_;
}
	
//==== member functions ====

void Clock::begin(){
	start_=std::clock();
}

void Clock::end(){
	stop_=std::clock();
}

double Clock::duration(){
	ticks_=stop_-start_;
	time_=((double)ticks_)/CLOCKS_PER_SEC;
	return time_;
}