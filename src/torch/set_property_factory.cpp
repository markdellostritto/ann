// c++
#include <stdexcept>
#include <iostream>
// str
#include "str/string.hpp"
// torch
#include "torch/set_property_factory.hpp"
#include "torch/set_property_list.hpp"

namespace property{

	std::shared_ptr<Base>& make(std::shared_ptr<Base>& base, const Name& name){
		switch(name){
			case Name::TYPE:{
				base.reset(new Type());
			}break;
			case Name::MASS:{
				base.reset(new Mass());
			}break;
			case Name::CHARGE:{
				base.reset(new Charge());
			}break;
			case Name::VELOCITY:{
				base.reset(new Velocity());
			}break;
			case Name::TEMP:{
				base.reset(new Temp());
			}break;
			default:{
				throw std::invalid_argument("comupte::state::make(std::shared_ptr<Base>&,compute::state::Name&): Invalid compute state name.");
			}break;
			return base;
		}
	}

	std::shared_ptr<Base>& read(std::shared_ptr<Base>& base, Token& token){
		const Name name=Name::read(string::to_upper(token.next()).c_str());
		switch(name){
			case Name::TYPE:{
				base.reset(new Type());
				static_cast<Type&>(*base).read(token);
			}break;
			case Name::MASS:{
				base.reset(new Mass());
				static_cast<Mass&>(*base).read(token);
			}break;
			case Name::CHARGE:{
				base.reset(new Charge());
				static_cast<Charge&>(*base).read(token);
			}break;
			case Name::VELOCITY:{
				base.reset(new Velocity());
				static_cast<Velocity&>(*base).read(token);
			}break;
			case Name::TEMP:{
				base.reset(new Temp());
				static_cast<Temp&>(*base).read(token);
			}break;
			default:{
				throw std::invalid_argument("compute::state::read(std::shared_ptr<Base>&,Token&): Invalid compute state name.");
			}break;
		}
		return base;
	}

	std::ostream& operator<<(std::ostream& out, const std::shared_ptr<Base>& base){
		switch(base->name()){
			case Name::TYPE: out<<static_cast<const Type&>(*base); break;
			case Name::MASS: out<<static_cast<const Mass&>(*base); break;
			case Name::CHARGE: out<<static_cast<const Charge&>(*base); break;
			case Name::VELOCITY: out<<static_cast<const Velocity&>(*base); break;
			case Name::TEMP: out<<static_cast<const Temp&>(*base); break;
			default: break;
		}
		return out;
	}
	
}
