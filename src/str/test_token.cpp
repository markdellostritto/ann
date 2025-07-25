// c++
#include <iostream>
// str
#include "str/print.hpp"
#include "str/token.hpp"
#include "str/string.hpp"

int main(int argc, char* argv[]){
	
	char* strbuf=new char[print::len_buf];
	Token token;
	
	const char* str_test_space="this is a test str";
	const char* str_test_ws="this is\ta\ntest\rstr";
	const char* str_test_other="this=is+a-test_str";
	
	//test - splitting string - space
	token.read(str_test_space," ");
	std::cout<<print::buf(strbuf)<<"\n";
	std::cout<<"test - split - space\n";
	std::cout<<print::buf(strbuf,'*')<<"\n";
	std::cout<<"test string = \n"<<str_test_space<<"\n";
	std::cout<<print::buf(strbuf,'*')<<"\n";
	std::cout<<"tokens = \n";
	while(!token.end()) std::cout<<token.next()<<"\n";
	std::cout<<print::buf(strbuf,'*')<<"\n";
	std::cout<<print::buf(strbuf)<<"\n";
	
	//test - splitting string - white space
	token.read(str_test_ws,string::WS);
	std::cout<<print::buf(strbuf)<<"\n";
	std::cout<<"test - split - white space\n";
	std::cout<<print::buf(strbuf,'*')<<"\n";
	std::cout<<"test string = \n"<<str_test_ws<<"\n";
	std::cout<<print::buf(strbuf,'*')<<"\n";
	std::cout<<"tokens = \n";
	while(!token.end()) std::cout<<token.next()<<"\n";
	std::cout<<print::buf(strbuf,'*')<<"\n";
	std::cout<<print::buf(strbuf)<<"\n";
	
	//test - splitting string - other
	token.read(str_test_other,"=+-_");
	std::cout<<print::buf(strbuf)<<"\n";
	std::cout<<"test - split - other\n";
	std::cout<<print::buf(strbuf,'*')<<"\n";
	std::cout<<"test string = \n"<<str_test_other<<"\n";
	std::cout<<print::buf(strbuf,'*')<<"\n";
	std::cout<<"tokens = \n";
	while(!token.end()) std::cout<<token.next()<<"\n";
	std::cout<<print::buf(strbuf,'*')<<"\n";
	std::cout<<print::buf(strbuf)<<"\n";
	
	//test - splitting string - other
	std::cout<<print::buf(strbuf)<<"\n";
	std::cout<<"test - split - multiple\n";
	std::cout<<print::buf(strbuf,'*')<<"\n";
	std::cout<<"test string = \n"<<str_test_space<<"\n";
	std::cout<<print::buf(strbuf,'*')<<"\n";
	token.read(str_test_space,string::WS);
	std::cout<<"token :-1 = "<<token.next(-1)<<"\n";
	token.read(str_test_space,string::WS);
	std::cout<<"token : 0 = "<<token.next(0)<<"\n";
	token.read(str_test_space,string::WS);
	std::cout<<"token : 1 = "<<token.next(1)<<"\n";
	token.read(str_test_space,string::WS);
	std::cout<<"token : 2 = "<<token.next(2)<<"\n";
	token.read(str_test_space,string::WS);
	std::cout<<"token : 3 = "<<token.next(3)<<"\n";
	token.read(str_test_space,string::WS);
	std::cout<<"token : 4 = "<<token.next(4)<<"\n";
	token.read(str_test_space,string::WS);
	std::cout<<"token : 5 = "<<token.next(5)<<"\n";
	try{
		token.read(str_test_space,string::WS);
		std::cout<<"token - 6 = "<<token.next(6)<<"\n";
	}catch(std::exception& e){
		std::cout<<e.what()<<"\n";
		std::cout<<"Could not find 6th token\n";
	}
	std::cout<<print::buf(strbuf)<<"\n";
	
	//test - splitting string - chained
	std::cout<<print::buf(strbuf)<<"\n";
	std::cout<<"test - split - space\n";
	std::cout<<print::buf(strbuf,'*')<<"\n";
	std::cout<<"test string = \n"<<str_test_space<<"\n";
	std::cout<<print::buf(strbuf,'*')<<"\n";
	std::cout<<"token : 3 = "<<token.read(str_test_space,string::WS).next(3)<<"\n";
	std::cout<<print::buf(strbuf,'*')<<"\n";
	std::cout<<print::buf(strbuf)<<"\n";
	
	delete[] strbuf;
	
	return 0;
}