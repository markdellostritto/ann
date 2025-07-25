// c libraries
#include <cstring>
#include <iostream>
#include <cctype>
// parse
#include "str/parse.hpp"

namespace input{
	
	void Arg::clear(){
		key_.clear();
		vals_.clear();
	}
	
	void parse(int argc, char* argv[], std::vector<Arg>& args){
		args.clear();
		//loop over all arguments
		for(int i=1; i<argc; ++i){
			//check for argument
			if(argv[i][0]=='-'){
				if(std::strlen(argv[i])==1) continue;//skip "-"
				if(std::isdigit(argv[i][1])) continue;//skip "-(number)"
				args.push_back(Arg());
				args.back().key()=std::string(argv[i]+1);
				for(int j=i+1; j<argc; ++j){
					//if we get to "-" which is not followed by a number, break
					if(argv[j][0]=='-'){
						if(std::strlen(argv[j])>1){
							if(!std::isdigit(argv[j][1])) break;
						} else break;
					}
					args.back().vals().push_back(argv[j]);
				}
			}
		}
	}
	
} //end namespace input
