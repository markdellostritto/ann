//c++
#include <iostream>
#include <algorithm>
// math
#include "math/special.hpp"
// analysis
#include "analysis/group.hpp"
// structure
#include "struc/structure.hpp"
// text
#include "str/string.hpp"

//***********************************************************************
// Group
//***********************************************************************

//==== Style ====

std::ostream& operator<<(std::ostream& out, const Group::Style& style){
	switch(style){
		case Group::Style::ID: out<<"ID"; break;
		case Group::Style::TYPE: out<<"TYPE"; break;
		case Group::Style::NAME: out<<"NAME"; break;
		default: out<<"UNKNOWN"; break;
	}
	return out;
}

Group::Style Group::Style::read(const char* str){
	if(std::strcmp(str,"ID")==0) return Group::Style::ID;
	else if(std::strcmp(str,"TYPE")==0) return Group::Style::TYPE;
	else if(std::strcmp(str,"NAME")==0) return Group::Style::NAME;
	else return Group::Style::UNKNOWN;
}

const char* Group::Style::name(const Group::Style& style){
	switch(style){
		case Group::Style::ID: return "ID";
		case Group::Style::TYPE: return "TYPE";
		case Group::Style::NAME: return "NAME";
		default: return "UNKNOWN";
	}
}

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Group& group){
	return out<<group.label();
}

//==== member functions ====

//modification

void Group::clear(){
	style_=Group::Style::UNKNOWN;
	label_="NULL";
	id_=string::hash("NULL");
	atoms_.clear();
	types_.clear();
	names_.clear();
	limits_.clear();
}

void Group::resize(int natoms){
	if(natoms<0) throw std::invalid_argument("Group::resize(int): Invalid group size.");
	atoms_.resize(natoms);
}

void Group::label(const char* str){
	label_=str;
	id_=string::hash(label_);
}

void Group::init(const std::string& label, const std::vector<int>& atoms){
	label_=label;
	id_=string::hash(label_);
	atoms_=atoms;
	std::sort(atoms_.begin(),atoms_.begin()+atoms_.size());
}

//atoms

bool Group::contains(int atom){
	for(int i=0; i<atoms_.size(); ++i){
		if(atom<=atoms_[i]){
			if(atom==atoms_[i]) return true;
			else return false;
		}
	}
	return false;
}

int Group::find(int atom){
	int index=-1;
	for(int i=0; i<atoms_.size(); ++i){
		if(atom==atoms_[i]){
			index=i; break;
		}
	}
	return index;
}

//reading/writing

void Group::read(Token& token){
	//read label
	label_=token.next();
	//read style
	style_=Group::Style::read(string::to_upper(token.next()).c_str());
	//read atoms
	switch(style_){
		case Group::Style::ID:{
			//note: zero-indexed
			limits_.clear();
			while(!token.end()){
				//split atom string
				std::string atomstr=token.next();
				Token atok(atomstr.c_str(),":");
				//read atom limits
				int beg=0,end=0,stride=1;
				beg=std::atoi(atok.next().c_str());
				if(!atok.end()) end=std::atoi(atok.next().c_str());
				else end=beg;
				if(!atok.end()) stride=std::atoi(atok.next().c_str());
				//check limits
				if(stride<=0) throw std::invalid_argument("Group::read(const Token&): Invalid atom stride.");
				//append
				std::array<int,3> arr;
				arr[0]=beg;
				arr[1]=end;
				arr[2]=stride;
				limits_.push_back(arr);
			}
		} break;
		case Group::Style::TYPE:{
			types_.clear();
			while(!token.end()){
				types_.push_back(std::atoi(token.next().c_str())-1);
			}
		} break;
		case Group::Style::NAME:{
			names_.clear();
			while(!token.end()){
				names_.push_back(token.next());
			}
		} break;
		default:
			throw std::invalid_argument("Group::read(Token&): Invalid group style.");
		break;
	}
}

//building

void Group::build(const Structure& struc){
	atoms_.clear();
	switch(style_){
		case Group::Style::ID:{
			//set atoms
			for(int i=0; i<limits_.size(); ++i){
				const int nAtoms=struc.nAtoms();
				const int nAtoms1=nAtoms+1;
				const int beg=(limits_[i][0]%nAtoms1+nAtoms1)%nAtoms1-1;
				const int end=(limits_[i][1]%nAtoms1+nAtoms1)%nAtoms1-1;
				const int stride=limits_[i][2];
				for(int j=beg; j<=end; j+=stride){
					atoms_.push_back(j);
				}
			}
		}break;
		case Group::Style::TYPE:{
			//find number of types
			int ntypes=-1;
			for(int i=0; i<struc.nAtoms(); ++i){
				if(struc.type(i)>ntypes) ntypes=struc.type(i);
			}
			ntypes++;
			//check types
			for(int i=0; i<types_.size(); ++i){
				if(types_[i]<0 || types_[i]>=ntypes) throw std::invalid_argument("Group::build(const Structure): Invalid type");
			}
			//set atoms
			for(int i=0; i<struc.nAtoms(); ++i){
				for(int j=0; j<types_.size(); ++j){
					if(struc.type(i)==types_[j]){
						atoms_.push_back(i);
						break;
					}
				}
			}
		} break;
		case Group::Style::NAME:{
			//set atoms
			for(int i=0; i<struc.nAtoms(); ++i){
				for(int j=0; j<names_.size(); ++j){
					if(struc.name(i)==names_[j]){
						atoms_.push_back(i);
						break;
					}
				}
			}
		} break;
		default:
			throw std::invalid_argument("Group::build(const Structure&): Invalid group style.");
		break;
	}
	std::sort(atoms_.begin(),atoms_.begin()+atoms_.size());
}
