#ifndef MOLECULE_HPP
#define MOLECULE_HPP

class Molecule{
public:
	class Atom{
	public:
		int an_;
		std::string name_;
		Eigen::Vector3d posn_;
	};
private:
	std::vector<Atom> atoms_;
public:
	//==== constructors/destructors ====
	Molecule(){}
	~Molecule(){}
	
	//==== access ====
	const int natoms(){return atoms_.size();}
	std::vector<Atom>& atoms(){return atoms_;}
	const std::vector<Atom>& atoms()const{return atoms_;}
};
