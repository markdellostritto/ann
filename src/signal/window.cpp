// c libraries
#include <cstring>
// c++ libraries
#include <ostream>
// ann - signal
#include "signal/window.hpp"

namespace window{
	
NAME::type NAME::read(const char* str){
	if(std::strcmp(str,"IDENTITY")==0) return NAME::IDENTITY;
	else if(std::strcmp(str,"BLACKMAN-HARRIS")==0) return NAME::BLACKMANHARRIS;
	else if(std::strcmp(str,"GAUSSIAN")==0) return NAME::GAUSSIAN;
	else return NAME::UNKNOWN;
}

std::ostream& operator<<(std::ostream& out, const NAME::type& t){
	switch(t){
		case NAME::IDENTITY: out<<"IDENTITY"; break;
		case NAME::BLACKMANHARRIS: out<<"BLACKMANHARRIS"; break;
		case NAME::GAUSSIAN: out<<"GAUSSIAN"; break;
		default: out<<"UNKNOWN";
	}
	return out;
}

}