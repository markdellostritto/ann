#ifndef POT_FACTORY_HPP
#define POT_FACTORY_HPP

// c++
#include <memory>
#include <iosfwd>
// str
#include "str/token.hpp"
// torch
#include "torch/pot.hpp"

#ifndef POT_FAC_FUNC
#define POT_FAC_FUNC 0
#endif

//factory

namespace ptnl{
	
	std::shared_ptr<Pot>& make(std::shared_ptr<Pot>& pot, const ptnl::Pot::Name& name);
	std::shared_ptr<Pot>& read(std::shared_ptr<Pot>& pot, Token& token);
	void coeff(std::vector<std::shared_ptr<Pot> > pots, Token& token);
	std::ostream& operator<<(std::ostream& out, const std::shared_ptr<Pot>& pot);
}

//serialization

namespace serialize{
	
	//**********************************************
	// byte measures
	//**********************************************
	
	template <> int nbytes(const std::shared_ptr<ptnl::Pot>& obj);
	
	
	//**********************************************
	// packing
	//**********************************************
	
	template <> int pack(const std::shared_ptr<ptnl::Pot>& obj, char* arr);
	
	//**********************************************
	// unpacking
	//**********************************************
	
	template <> int unpack(std::shared_ptr<ptnl::Pot>& obj, const char* arr);
	
}

#endif