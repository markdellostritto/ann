#ifndef NETWORK_HPP
#define NETWORK_HPP

// c++
#include <vector>
#include <iosfwd>

template <typename T>
Network<T>{
private:
	int size_;
	std::vector<T> nodes_;
	Eigen::VectorXi color_;
	Eigen::MatrixXi amat_;
public:
	//==== constructors/destructors ====
	Network(){}
	Network(int size){resize(size_);}
	~Network(){}
	
	//==== member access ====
	//size
	const int& size()const{return size_;}
	//nodes
	const std::vector<T>& nodes()const{return nodes_;}
	T& node(int i){return nodes_[i];}
	const T& node(int i)const{return nodes_[i];}
	//colors
	int& color(int i){return color_[i];}
	const int& color(int i)const{return color_[i];}
	//adjacency matrix
	const Eigen::MatrixXi& amat()const{return amat_;}
	int& edge(int i, int j){return amat_(i,j);}
	const int& edge(int i, int j)const{return amat_(i,j);}
	
	//==== member functions ====
	void clear();
	void resize(int n);
	
	//==== static functions ====
	static void color_set(Graph& graph, int color);
	static int color_cc(Graph& graph);
}

//==== member functions ====

template <typename T>
void Network<T>::clear(){
	nodes_.clear();
	color_.clear();
	amat_.setZero();
}

template <typename T>
void Network<T>::resize(int n){
	nodes_.resize(n);
	color_=Eigen::MatrixXi::Constant(n,-1);
	amat_=Eigen::MatrixXi::Zero(n,n);
}

//==== static functions ====

/**
* Set the color for every node in the graph
* @param graph - graph for which we set the color
* @param color - color which each node will have at the end
*/
template <typename T>
void Network<T>::color_set(Network<T>& network, int color){
	for(int i=0; i<network.size(); ++i){
		network.color(i)=color;
	}
}

/**
* Color a graph such that all nodes within the same connected component will
* have the same color.
* @param graph - graph for which we set the color
*/
template <typename T>
int Network<T>::color_cc(Network<T>& network){
	//reset graph color
	Network<T>::color_set(network,-1);
	int color=0;
	const int size=network.size();
	//loop over all nodes
	for(int i=0; i<size; ++i){
		//check if colored
		if(network.color(i)<0){
			//iterate over all connected nodes
			std::queue<int> nqueue;
			nqueue.push(i);
			while(nqueue.size()>0){
				//pop current node
				const int j=nqueue.front();
				nqueue.pop();
				if(network.color(j)<0){
					//color current node
					network.color(j)=color;
					//add all uncolored nodes to the queue
					for(int k=0; k<size; ++k){
						if(network.edge(j,k)>0 && network.color(k)<0){
							nqueue.push(k);
						}
					}
				}
			}
			//increment color
			color++;
		}
	}
	//return the number of cc's
	return color;
}


#endif