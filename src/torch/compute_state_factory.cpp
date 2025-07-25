// c++
#include <stdexcept>
// str
#include "str/string.hpp"
// torch
#include "torch/compute_state_factory.hpp"
#include "torch/compute_state_list.hpp"

namespace compute{

namespace state{

	std::shared_ptr<Base>& make(std::shared_ptr<Base>& base, const Name& name){
		switch(name){
			case Name::KE:{
				base.reset(new KE());
			}break;
			case Name::PE:{
				base.reset(new PE());
			}break;
			case Name::TE:{
				base.reset(new TE());
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
			case Name::KE:{
				base.reset(new KE());
				static_cast<KE&>(*base).read(token);
			}break;
			case Name::PE:{
				base.reset(new PE());
				static_cast<PE&>(*base).read(token);
			}break;
			case Name::TE:{
				base.reset(new TE());
				static_cast<TE&>(*base).read(token);
			}break;
			case Name::TEMP:{
				base.reset(new Temp());
				static_cast<Temp&>(*base).read(token);
			}break;
			default:{
				throw std::invalid_argument("compute::state::read(std::shared_ptr<Base>&,Token&): Invalid compute state name.");
			}break;
		}
	}

	std::ostream& operator<<(std::ostream& out, const std::shared_ptr<Base>& base){
		switch(base->name()){
			case Name::KE: out<<static_cast<const KE&>(*base); break;
			case Name::PE: out<<static_cast<const PE&>(*base); break;
			case Name::TE: out<<static_cast<const TE&>(*base); break;
			case Name::TEMP: out<<static_cast<const Temp&>(*base); break;
			default: break;
		}
		return out;
	}
	
}

}
