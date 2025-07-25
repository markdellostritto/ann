// c++
#include <algorithm>
#include <iostream>
// string
#include "str/string.hpp"
// chem
#include "chem/molkit.hpp"
#include "chem/ptable.hpp"

namespace molkit{

//***************************************************************
// Type
//***************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Type& type){
	return out<<type.element()<<" "<<type.label_<<" "<<type.element_<<" "<<type.lsmiles_name_<<" ("<<type.lsmiles_hash_<<")";
}
	
//==== static functions ====

Type& Type::read(Token& token, Type& type){
	type.label()=token.next();
	type.element()=token.next();
	type.an()=ptable::an(type.element().c_str());
	if(!token.end()){
		type.lsmiles_name()=token.next();
		type.lsmiles_hash()=string::hash(type.lsmiles_name().c_str());
	} else {
		type.lsmiles_name().clear();
		
	}
	return type;
}

//***************************************************************
// Alias
//***************************************************************

std::ostream& operator<<(std::ostream& out, const Alias& alias){
	out<<alias.alias()<<" ";
	for(int i=0; i<alias.labels().size(); ++i){
		out<<alias.labels()[i]<<" ";
	}
	return out;
}

Alias& Alias::read(Token& token, Alias& alias){
	alias.clear();
	alias.alias()=token.next();
	while(!token.end()){
		alias.labels().push_back(token.next());
	}
	return alias;
}

//***************************************************************
// Coeff
//***************************************************************

std::ostream& operator<<(std::ostream& out, const Coeff& coeff){
	out<<coeff.type()<<" ";
	for(int i=0; i<coeff.params().size(); ++i){
		out<<coeff.params()[i]<<" ";
	}
	return out;
}

Coeff& Coeff::read(Token& token, Coeff& coeff){
	coeff.clear();
	coeff.type()=std::atoi(token.next().c_str());
	while(!token.end()){
		coeff.params().push_back(std::atof(token.next().c_str()));
	}
	return coeff;
}

//***************************************************************
// Link
//***************************************************************

std::ostream& operator<<(std::ostream& out, const Link& link){
	out<<link.type()<<" ";
	for(int i=0; i<link.labels().size(); ++i){
		out<<link.labels()[i]<<" ";
	}
	return out;
}

Link& Link::read(Token& token, Link& link){
	link.clear();
	link.type()=std::atoi(token.next().c_str());
	while(!token.end()){
		link.labels().push_back(token.next());
	}
	return link;
}

//***************************************************************
// Molecule Functions
//***************************************************************

Structure& remove_H(const Structure& struc1, Structure& struc2){
	struc2.clear();
	int nAtomM=0;
	for(int i=0; i<struc1.nAtoms(); ++i){
		if(struc1.name(i)!="H") nAtomM++;
	}
	AtomType atomT=struc1.atomType();
	struc2.init(struc1.R());
	struc2.resize(nAtomM,atomT);
	int c=0;
	for(int i=0; i<struc1.nAtoms(); ++i){
		if(struc1.name(i)!="H"){
			if(atomT.name) struc2.name(c)=struc1.name(i);
			if(atomT.an) struc2.an(c)=struc1.an(i);
			if(atomT.type) struc2.type(c)=struc1.type(i);
			if(atomT.index) struc2.index(c)=struc1.index(i);
			if(atomT.mass) struc2.mass(c)=struc1.mass(i);
			if(atomT.charge) struc2.charge(c)=struc1.charge(i);
			if(atomT.posn) struc2.posn(c)=struc1.posn(i);
			if(atomT.force) struc2.force(c)=struc1.force(i);
			if(atomT.vel) struc2.vel(c)=struc1.vel(i);
			c++;
		}
	}
	return struc2;
}

Graph& make_graph(const Structure& struc, Graph& graph){
	Eigen::Vector3d tmp;
	//resize graph
	graph.clear();
	graph.resize(struc.nAtoms());
	//set bonds (edges)
	for(int i=0; i<struc.nAtoms(); ++i){
		for(int j=i+1; j<struc.nAtoms(); ++j){
			const double dr2=struc.dist2(struc.posn(i),struc.posn(j),tmp);
			const double bl=(struc.radius(i)+struc.radius(j));
			if(dr2<bl*bl) graph.push_edge(i,j);
		}
	}
	return graph;
}

std::vector<int>& make_lsmiles(const Structure& struc, Graph& graph, int atom, int depth, std::vector<int>& lsmiles){
	//color the graph
	graph.color_path(graph,atom,depth);
	//build the local smiles string
	std::vector<std::vector<int> > lsmilesv(depth);
	for(int i=0; i<graph.size(); ++i){
		if(graph.node(i).color()>0){
			lsmilesv[graph.node(i).color()-1].push_back(struc.an(i));
		}
	}
	//sort the smiles strings
	for(int i=0; i<depth; ++i){
		std::sort(lsmilesv[i].begin(),lsmilesv[i].end());
	}
	//pack the string
	lsmiles.clear();
	for(int i=0; i<depth; ++i){
		for(int j=0; j<lsmilesv[i].size(); ++j){
			lsmiles.push_back(lsmilesv[i][j]);
		}
	}
	//return 
	return lsmiles;
}
	
};