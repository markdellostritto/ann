// c++
#include <algorithm>
#include <iostream>
// string
#include "str/string.hpp"
// chem
#include "chem/molkit2.hpp"
#include "chem/ptable.hpp"

namespace molkit{

//***************************************************************
// Type
//***************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Type& type){
	return out<<type.name_<<" "<<type.depth_<<" "<<type.lsmiles_;
}
	
//==== static functions ====

Type& Type::read(Token& token, Type& type){
	type.name()=token.next();
	type.depth()=std::atoi(token.next().c_str());
	type.lsmiles()=token.next();
	return type;
}

bool operator==(const Type& type1, const Type& type2){
	return (type1.depth()==type2.depth() && type1.lsmiles()==type2.lsmiles());
}

//***************************************************************
// Label
//***************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Label& label){
	out<<"index "<<label.index_<<" names ";
	for(int i=0; i<label.types().size(); ++i){
		out<<label.type(i).name()<<"("<<label.type(i).depth()<<") ";
	}
	return out;
}

//==== member functions ====

void Label::clear(){
	types_.clear();
	index_=-1;
}

//***************************************************************
// Coeff
//***************************************************************

//==== operators ====

std::ostream& operator<<(std::ostream& out, const Coeff& coeff){
	out<<"index "<<coeff.index_<<" coeffs ";
	for(int i=0; i<coeff.params().size(); ++i){
		out<<coeff.param(i)<<" ";
	}
	return out;
}

//==== member functions ====

void Coeff::clear(){
	params_.clear();
	index_=-1;
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
	const int nAtoms=struc.nAtoms();
	//resize graph
	graph.clear();
	graph.resize(nAtoms);
	//set bonds (edges)
	for(int i=0; i<nAtoms; ++i){
		if(i%1000==0) std::cout<<"atom "<<i<<" ("<<nAtoms<<")\n";
		const Eigen::Vector3d posn=struc.posn(i);
		const double ri=struc.radius(i);
		for(int j=i+1; j<nAtoms; ++j){
			const double dr2=struc.dist2(posn,struc.posn(j),tmp);
			const double bl=(ri+struc.radius(j));
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