#pragma once
#ifndef SET_PROPERTY_FACTORY_HPP
#define SET_PROPERTY_FACTORY_HPP

// c++
#include <memory>
#include <iosfwd>
// torch
#include "torch/set_property.hpp"

namespace property{

	std::shared_ptr<Base>& make(std::shared_ptr<Base>& base, const Name& name);
	std::shared_ptr<Base>& read(std::shared_ptr<Base>& base, Token& token);
	std::ostream& operator<<(std::ostream& out, const std::shared_ptr<Base>& base);
	
}

#endif