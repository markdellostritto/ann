// c++
#include <queue>
#include <stdexcept>
#include <iostream>
// eigen
#include <Eigen/Dense>
// math
#include "math/graph.hpp"

//==== member functions ====

void Graph::clear(){
	nodes_.clear();
	edges_.clear();
}

void Graph::resize(int n){
	size_=n;
	nodes_.resize(n);
	edges_.resize(n);
}

void Graph::push_edge(int i, int j){
	if(i<0 || i>=size_) throw std::invalid_argument("Graph::push_edge(int,int): invalid beg node.");
	if(j<0 || j>=size_) throw std::invalid_argument("Graph::push_edge(int,int): invalid end node.");
	bool match=false;
	for(int n=0; n<edges_[i].size(); ++n){
		if(edges_[i][n].end()==j){
			match=true;
			break;
		}
	}
	if(!match){
		edges_[i].push_back(Edge(i,j));
		edges_[j].push_back(Edge(j,i));
	}
}

void Graph::pull_edge(int i, int j){
	if(i<0 || i>=size_) throw std::invalid_argument("Graph::pull_edge(int,int): invalid beg node.");
	if(j<0 || j>=size_) throw std::invalid_argument("Graph::pull_edge(int,int): invalid end node.");
	for(int n=0; n<edges_[i].size(); ++n){
		if(edges_[i][n].end()==j){
			edges_[i][n]=edges_[i].back();
			edges_[i].pop_back();
			break;
		}
	}
	for(int n=0; n<edges_[j].size(); ++n){
		if(edges_[j][n].end()==i){
			edges_[j][n]=edges_[j].back();
			edges_[j].pop_back();
			break;
		}
	}
}

//==== static functions ====

/**
* Set the color for every node in the graph
* @param graph - graph for which we set the color
* @param color - color which each node will have at the end
*/
void Graph::color_set(Graph& graph, int color){
	for(int i=0; i<graph.size(); ++i){
		graph.node(i).color()=color;
	}
}

/**
* Color a graph such that the color is the distance from the head node
* up to a given depth (-1 if past that depth).  The head node will be "0".
* @param graph - graph for which we set the color
* @param head - the head node for the start of the path
* @param depth - the max distance from the head node for which we will color nodes
*/
void Graph::color_path(Graph& graph, int head, int depth){
	if(head<0 || head>=graph.size()) throw std::invalid_argument("Graph::color_path(Graph&,int): invalid head node.");
	if(depth<0) depth=graph.size();
	//reset graph color
	Graph::color_set(graph,-1);
	//add head to queue
	std::queue<int> nqueue;
	nqueue.push(head);
	//color all pathways from "head"
	graph.node(head).color()=0;
	while(nqueue.size()>0){
		//retrieve current node
		const int i=nqueue.front();
		nqueue.pop();
		if(graph.node(i).color()<depth){
			//find all neighbors (not colored)
			for(int j=0; j<graph.edges(i).size(); ++j){
				const int nn=graph.edge(i,j).end();
				if(graph.node(nn).color()<0){
					//color node
					graph.node(nn).color()=graph.node(i).color()+1;
					//add node to queue
					nqueue.push(nn);
				}
			}
		}
	}
}

/**
* Color a graph such that all nodes within the same connected component will
* have the same color.
* @param graph - graph for which we set the color
*/
int Graph::color_cc(Graph& graph){
	//reset graph color
	Graph::color_set(graph,-1);
	int color=0;
	//loop over all nodes
	for(int i=0; i<graph.size(); ++i){
		//check if colored
		if(graph.node(i).color()<0){
			//iterate over all connected nodes
			std::queue<int> nqueue;
			nqueue.push(i);
			while(nqueue.size()>0){
				//pop current node
				const int j=nqueue.front();
				nqueue.pop();
				if(graph.node(j).color()<0){
					//color current node
					graph.node(j).color()=color;
					//add all uncolored nodes to the queue
					for(int k=0; k<graph.edges(j).size(); ++k){
						const int nn=graph.edge(j,k).end();
						if(graph.node(nn).color()<0) nqueue.push(nn);
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

/**
* Create an adjacency matrix from a given graph.
* @param graph - graph for which we will generate a matrix
* @param mat - adjacency matrix
*/
Eigen::MatrixXi& Graph::adj_mat(Graph& graph, Eigen::MatrixXi& mat){
	mat=Eigen::MatrixXi::Zero(graph.size(),graph.size());
	for(int i=0; i<graph.size(); ++i){
		for(int j=0; j<graph.edges(i).size(); ++j){
			mat(i,graph.edge(i,j).end())++;
		}
	}
	return mat;
}

/**
* Combine two graphs into one, keeping all original connections, but resulting
* in a graph where each subgraph is a separate connected component
* @param graph1 - first graph
* @param graph2 - second graph
* @param graph - united graph
*/
void Graph::unite(const Graph& graph1, const Graph& graph2, Graph& graph){
	//resize the graph
	const int size1=graph1.size();
	const int size2=graph2.size();
	const int size=size1+size2;
	graph.clear();
	graph.resize(size);
	//add the nodes
	for(int i=0; i<size1; ++i){
		graph.node(i)=graph1.node(i);
	}
	for(int i=0; i<size2; ++i){
		graph.node(i+size1)=graph2.node(i);
	}
	//add the edges
	for(int i=0; i<size1; ++i){
		for(int j=0; j<graph1.edges(i).size(); ++j){
			graph.push_edge(graph1.edge(i,j).beg(),graph1.edge(i,j).end());
		}
	}
	for(int i=0; i<size2; ++i){
		for(int j=0; j<graph2.edges(i).size(); ++j){
			graph.push_edge(graph2.edge(i,j).beg()+size1,graph2.edge(i,j).end()+size1);
		}
	}
}

//==== generation functions ====

Graph& Graph::make_line(Graph& graph, int n){
	if(n<=0) throw std::invalid_argument("Graph::make_line(Graph&,int): invalid size.");
	graph.clear();
	graph.resize(n);
	for(int i=0; i<n-1; ++i){
		graph.push_edge(i,i+1);
	}
	return graph;
}

Graph& Graph::make_cycle(Graph& graph, int n){
	if(n<=0) throw std::invalid_argument("Graph::make_cycle(Graph&,int): invalid size.");
	graph.clear();
	graph.resize(n);
	for(int i=0; i<n-1; ++i){
		graph.push_edge(i,i+1);
	}
	graph.push_edge(n-1,0);
	return graph;
}

Graph& Graph::make_fc(Graph& graph, int n){
	if(n<=0) throw std::invalid_argument("Graph::make_fc(Graph&,int): invalid size.");
	graph.clear();
	graph.resize(n);
	for(int i=0; i<n; ++i){
		for(int j=i; j<n; ++j){
			graph.push_edge(i,j);
		}
	}
	return graph;
}