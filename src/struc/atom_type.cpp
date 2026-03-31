//c libraries
#include <cstring>
//c++ libraries
#include <iostream>
// structure
#include "struc/atom_type.hpp"
// str
#include "str/string.hpp"

//**********************************************************************************************
//AtomType
//**********************************************************************************************

std::ostream& operator<<(std::ostream& out, const AtomType& atomT){
	//basic properties
	if(atomT.name)   out<<"name ";
	if(atomT.an)     out<<"an ";
	if(atomT.type)   out<<"type ";
	if(atomT.index)  out<<"index ";
	//serial properties
	if(atomT.mass)   out<<"mass ";
	if(atomT.charge) out<<"charge ";
	if(atomT.radius) out<<"radius ";
	if(atomT.entropy)out<<"entropy ";
	if(atomT.chi)    out<<"chi ";
	if(atomT.eta)    out<<"eta ";
	if(atomT.c6)     out<<"c6 ";
	if(atomT.js)     out<<"js ";
	if(atomT.alpha)  out<<"alpha ";
	if(atomT.weight) out<<"weight ";
	if(atomT.drudeQ) out<<"drudeQ ";
	if(atomT.drudeM) out<<"drudeM ";
	if(atomT.drudeW) out<<"drudeW ";
	if(atomT.drudeN) out<<"drudeN ";
	//vector properties
	if(atomT.image)  out<<"image ";
	if(atomT.posn)   out<<"posn ";
	if(atomT.vel)    out<<"vel ";
	if(atomT.force)  out<<"force ";
	if(atomT.spin)   out<<"spin ";
	if(atomT.drudeR) out<<"drudeR ";
	//nnp
	if(atomT.symm)		out<<"symm ";
	return out;
}

void AtomType::defaults(){
	//basic properties
	name   = false;
	an     = false;
	type   = false;
	index  = false;
	//serial properties
	mass   = false;
	charge = false;
	radius = false;
	entropy= false;
	chi    = false;
	eta    = false;
	c6     = false;
	js     = false;
	alpha  = false;
	weight = false;
	drudeQ = false;
	drudeM = false;
	drudeW = false;
	drudeN = false;
	//vector properties
	image  = false;
	posn   = false;
	vel    = false;
	force  = false;
	spin   = false;
	drudeR = false;
	//nnp
	symm   = false;
}

AtomType AtomType::read(Token& token){
	AtomType atomT;
	while(!token.end()){
		const std::string tag=string::to_upper(token.next());
		if(tag=="NAME") atomT.name=true;
		else if(tag=="AN") atomT.an=true;
		else if(tag=="TYPE") atomT.type=true;
		else if(tag=="INDEX") atomT.index=true;
		else if(tag=="MASS") atomT.mass=true;
		else if(tag=="CHARGE") atomT.charge=true;
		else if(tag=="RADIUS") atomT.radius=true;
		else if(tag=="ENTROPY") atomT.entropy=true;
		else if(tag=="CHI") atomT.chi=true;
		else if(tag=="ETA") atomT.eta=true;
		else if(tag=="C6") atomT.c6=true;
		else if(tag=="JS") atomT.js=true;
		else if(tag=="ALPHA") atomT.alpha=true;
		else if(tag=="WEIGHT") atomT.weight=true;
		else if(tag=="DRUDEQ") atomT.drudeQ=true;
		else if(tag=="DRUDEM") atomT.drudeM=true;
		else if(tag=="DRUDEW") atomT.drudeW=true;
		else if(tag=="DRUDEN") atomT.drudeN=true;
		else if(tag=="IMAGE") atomT.image=true;
		else if(tag=="POSN") atomT.posn=true;
		else if(tag=="VEL") atomT.vel=true;
		else if(tag=="FORCE") atomT.force=true;
		else if(tag=="SPIN") atomT.spin=true;
		else if(tag=="DRUDER") atomT.drudeR=true;
		else if(tag=="SYMM") atomT.symm=true;
	}
	return atomT;
}

//**********************************************************************************************
// serialization
//**********************************************************************************************

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const AtomType& obj){
		int size=0;
		//basic properties
		size+=sizeof(obj.name);
		size+=sizeof(obj.an);
		size+=sizeof(obj.type);
		size+=sizeof(obj.index);
		//serial properties
		size+=sizeof(obj.mass);
		size+=sizeof(obj.charge);
		size+=sizeof(obj.radius);
		size+=sizeof(obj.entropy);
		size+=sizeof(obj.chi);
		size+=sizeof(obj.eta);
		size+=sizeof(obj.c6);
		size+=sizeof(obj.js);
		size+=sizeof(obj.alpha);
		size+=sizeof(obj.weight);
		size+=sizeof(obj.drudeQ);
		size+=sizeof(obj.drudeM);
		size+=sizeof(obj.drudeW);
		size+=sizeof(obj.drudeN);
		//vector properties
		size+=sizeof(obj.image);
		size+=sizeof(obj.posn);
		size+=sizeof(obj.vel);
		size+=sizeof(obj.force);
		size+=sizeof(obj.spin);
		size+=sizeof(obj.drudeR);
		//nnp
		size+=sizeof(obj.symm);
		return size;
	}
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const AtomType& obj, char* arr){
		int pos=0;
		//basic properties
		std::memcpy(arr+pos,&obj.name,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.an,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.type,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.index,sizeof(bool)); pos+=sizeof(bool);
		//serial properties
		std::memcpy(arr+pos,&obj.mass,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.charge,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.radius,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.entropy,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.chi,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.eta,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.c6,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.js,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.alpha,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.weight,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.drudeQ,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.drudeM,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.drudeW,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.drudeN,sizeof(bool)); pos+=sizeof(bool);
		//vector properties
		std::memcpy(arr+pos,&obj.image,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.posn,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.vel,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.force,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.spin,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(arr+pos,&obj.drudeR,sizeof(bool)); pos+=sizeof(bool);
		//nnp
		std::memcpy(arr+pos,&obj.symm,sizeof(bool)); pos+=sizeof(bool);
		return pos;
	}
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(AtomType& obj, const char* arr){
		int pos=0;
		//basic properties
		std::memcpy(&obj.name,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.an,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.type,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.index,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		//serial properties
		std::memcpy(&obj.mass,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.charge,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.radius,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.entropy,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.chi,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.eta,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.c6,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.js,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.alpha,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.weight,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.drudeQ,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.drudeM,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.drudeW,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.drudeN,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		//vector properties
		std::memcpy(&obj.image,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.posn,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.vel,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.force,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.spin,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		std::memcpy(&obj.drudeR,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		//nnp
		std::memcpy(&obj.symm,arr+pos,sizeof(bool)); pos+=sizeof(bool);
		return pos;
	}
	
}
