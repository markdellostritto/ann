#ifndef LIST_HPP
#define LIST_HPP

#include <cmath>
#include <vector>

namespace list{

namespace sort{
	
	//***************************************************************
	//Sorting: Bubble
	//***************************************************************
	
	template <class T>
	std::vector<T>& bubble(std::vector<T>& arr){
		bool swap;
		if(arr.size()>1){
			do{
				swap=false;
				for(int i=0; i<arr.size()-1; ++i){
					if(arr[i+1]<arr[i]){
						T temp=arr[i];
						arr[i]=arr[i+1];
						arr[i+1]=temp;
						swap=true;
					}
				}
			}while(swap==true);
		}
		
		return arr;
	}
	
	//***************************************************************
	//Sorting: Insertion
	//***************************************************************
	
	template <class T>
	std::vector<T>& insertion(std::vector<T>& arr){
		for(int i=1; i<arr.size(); ++i){
			const T x=arr[i];
			int j=i-1;
			while(j>=0 && arr[j]>x){
				arr[j+1]=arr[j];
				--j;
			}
			arr[j+1]=x;
		}
		return arr;
	}
	
	template <class T>
	std::vector<T>& insertion(std::vector<T>& arr, std::vector<T>& val){
		for(int i=1; i<arr.size(); ++i){
			const T x=arr[i];
			const T y=val[i];
			int j=i-1;
			while(j>=0 && arr[j]>x){
				arr[j+1]=arr[j];
				arr[j+1]=arr[j];
				--j;
			}
			arr[j+1]=x;
			arr[j+1]=y;
		}
		return arr;
	}
	
	//***************************************************************
	//Sorting: Shell
	//***************************************************************
	
	template <class T>
	std::vector<T>& shell(std::vector<T>& arr){
		//find the maximum exponent of the power series of gaps by recursively generating the largest possible gap
		//the maximum gap should be size/3, rounded up [1]
		int max=arr.size()/3+((arr.size()%3)+1)%2;//this gets us arr/3, rounded up
		int pow=1,gap=1,powi=3;
		while(gap<max){
			gap=gap*3+1;
			++pow;
			powi*=3;
		}
		
		//now that we have the maximum exponenet, perform shell sorting
		for(int i=pow; i>=1; --i){
			gap=(powi-1)/2;
			powi/=3;
			for(int j=gap; j<arr.size(); j+=gap){
				const T temp=arr[j];
				int k=j;
				while(k>0 && arr[k-gap]>temp){
					arr[k]=arr[k-gap];
					k-=gap;
				}
				arr[k]=temp;
			}
		}
		
		return arr;
	}
}

namespace search{
	
	//***************************************************************
	//Searching
	//***************************************************************
	
	//min/max
	
	template <class T>
	T min(const std::vector<T>& arr){
		T min=0;
		for(int i=0; i<arr.size(); ++i) if(min>arr[i]) min=arr[i];
		return min;
	}
	
	template <class T>
	T max(const std::vector<T>& arr){
		T max=0;
		for(int i=0; i<arr.size(); ++i) if(max<arr[i]) max=arr[i];
		return max;
	}
	
	//min/max index
	
	template <class T>
	int minIndex(const std::vector<T>& arr){
		T min=0;
		int n=0;
		for(int i=0; i<arr.size(); ++i) if(min>arr[i]){min=arr[i];n=i;}
		return n;
	}
	
	template <class T>
	int maxIndex(const std::vector<T>& arr){
		T max=0;
		int n=0;
		for(int i=0; i<arr.size(); ++i) if(max<arr[i]){max=arr[i];n=i;};
		return n;
	}
	
	//searching - linear 
	
	template <class T>
	int find_exact_lin(const T& x, std::vector<T>& arr){
		for(int i=0; i<arr.size(); i++) if(x==arr[i]) return i;
		return -1;
	}
	
	template <class T>
	int find_approx_lin(const T& x, std::vector<T>& arr){
		for(int i=0; i<arr.size(); i++) if(std::fabs(x-arr[i])<1e-6) return i;
		return -1;
	}
	
	/*References:
		[1] Knuth, Donald E. (1997). "Shell's method". The Art of Computer Programming. 
			Volume 3: Sorting and Searching (2nd ed.). Reading, Massachusetts: Addison-Wesley. 
			pp. 83–95. ISBN 0-201-89685-0.
	*/
}

namespace search_ordered{
	
	template <class T> T min(const std::vector<T>& arr){return arr[0];};
	template <class T> T max(const std::vector<T>& arr){return arr.back();};
	
	template <class T>
	int find_exact(const T& x, const std::vector<T>& arr){
		int uLim=arr.size();//upper limit for Newton's method
		int lLim=0;//lower limit for Newton's method
		int mid;//middle point for Newton's method
		while(uLim-lLim>1){
			mid=lLim+(uLim-lLim)/2;
			if(arr[lLim]<=x && x<=arr[mid]) uLim=mid;
			else lLim=mid;
		}
		if(arr[lLim]==x) return lLim;
		else if(arr[uLim]==x) return uLim;
		else return -1;
	}
	
	template <class T>
	int find_approx(const T& x, const std::vector<T>& arr){
		int uLim=arr.size();//upper limit for Newton's method
		int lLim=0;//lower limit for Newton's method
		int mid;//middle point for Newton's method
		while(uLim-lLim>1){
			mid=lLim+(uLim-lLim)/2;
			if(arr[lLim]<=x && x<=arr[mid]) uLim=mid;
			else lLim=mid;
		}
		return lLim;
	}
}
}

#endif