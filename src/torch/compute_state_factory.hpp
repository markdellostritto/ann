#pragma once
#ifndef COMPUTE_STATE_FACTORY_HPP
#define COMPUTE_STATE_FACTORY_HPP

#include "torch/compute_state.hpp"

namespace compute{

namespace state{

	std::shared_ptr<Base>& make(std::shared_ptr<Base>& base, const Name& name);
	std::shared_ptr<Base>& read(std::shared_ptr<Base>& base, Token& token);
	std::ostream& operator<<(std::ostream& out, const std::shared_ptr<Base>& base);
	
}

}

#endif