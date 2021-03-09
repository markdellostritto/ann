#pragma once
#ifndef TIME_HPP
#define TIME_HPP

// c libraries
#include <ctime>

struct Clock{
private:
	clock_t start_,stop_,ticks_;
	double time_;
public:
	//==== constructors/destructors ====
	Clock():start_(0),stop_(0),time_(0){}
	~Clock(){}
	
	//==== operator ====
	friend std::ostream& operator<<(std::ostream& out, const Clock& c);
	
	//==== access ====
	clock_t& start(){return start_;}
	const clock_t& start()const{return start_;}
	clock_t& stop(){return stop_;}
	const clock_t& stop()const{return stop_;}
	clock_t& ticks(){return ticks_;}
	const clock_t& ticks()const{return ticks_;}
	double& time(){return time_;}
	const double& time()const{return time_;}
	
	//==== member functions ====
	void begin();
	void end();
	double duration();
	
};

#endif
