#ifndef GRAPH_HPP
#define GRAPH_HPP

// c++
#include <vector>
#include <iosfwd>
// 

//************************************************************
// Node
//************************************************************

class Node{
private:
	int color_;
public:
	//==== constructors/destructors ====
	Node():color_(-1){}
	~Node(){}
	
	//==== member access ====
	int& color(){return color_;}
	const int& color()const{return color_;}
};

//************************************************************
// Edge
//************************************************************

class Edge{
private:
	int beg_;
	int end_;
public:
	//==== constructors/destructors ====
	Edge():beg_(-1),end_(-1){}
	Edge(int beg, int end):beg_(beg),end_(end){}
	~Edge(){}
	
	//==== member access ====
	int& beg(){return beg_;}
	const int& beg()const{return beg_;}
	int& end(){return end_;}
	const int& end()const{return end_;}
};

//************************************************************
// Graph
//************************************************************

class Graph{
private:
	int size_;
	std::vector<Node> nodes_;
	std::vector<std::vector<Edge> > edges_;
public:
	//==== constructors/destructors ====
	Graph(){}
	Graph(int size){resize(size);}
	~Graph(){}
	
	//==== member access ====
	int size()const{return size_;}
	Node& node(int i){return nodes_[i];}
	const Node& node(int i)const{return nodes_[i];}
	Edge& edge(int i, int j){return edges_[i][j];}
	const Edge& edge(int i, int j)const{return edges_[i][j];}
	const std::vector<Edge>& edges(int i)const{return edges_[i];}
	
	//==== member functions ====
	void clear();
	void resize(int n);
	void push_edge(int i, int j);
	void pull_edge(int i, int j);	
	
	//==== static functions ====
	static void color_set(Graph& graph, int color);
	static void color_path(Graph& graph, int head, int depth=-1);
	static int color_cc(Graph& graph);
	static Eigen::MatrixXi& adj_mat(Graph& graph, Eigen::MatrixXi& mat);
	static void unite(const Graph& graph1, const Graph& graph2, Graph& graph);
	
	//==== generation functions ====
	static Graph& make_line(Graph& graph, int n);
	static Graph& make_cycle(Graph& graph, int n);
	static Graph& make_fc(Graph& graph, int n);
};

#endif
