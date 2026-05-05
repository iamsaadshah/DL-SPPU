#include<iostream>
#include<vector>
#include<omp.h>
#include<cstdlib>
using namespace std;

int main(){
    int n =1000000000;
    vector<int> nums(n);
    
    for(int i=0; i<n ;i++){
        nums[i]=rand()%10000;
    }
    
    cout<<"Input: "<<n<<" random integer from (0-9999)\n\n";
    
    // cout<<"\n the array is:\n [";
    // for(int i=0;i<40;i++){
    //     cout<<nums[i]<<" ";
    // }
    // cout<<']';
    
    // Declaration
    
    long long sum_seq=0, sum_par=0;
    int min_seq, max_seq;
    int min_par, max_par;
    double avg_seq, avg_par;
    double start, end;
    
    // Sequential
    
    min_seq = max_seq = nums[0];
    sum_seq = 0;
    
    start = omp_get_wtime();
    for(int i =0 ;i<n ;i++){
        if(nums[i] < min_seq) min_seq = nums[i];
        if(nums[i] > max_seq) max_seq = nums[i];
        sum_seq += nums[i];
    }
    
    end = omp_get_wtime();
    
    avg_seq = (double)sum_seq/n;
    double time_seq = end - start;
    // end Sequential
    
    // Parallel 
    min_par = max_par = nums[0];
    sum_par = 0;
    
    start = omp_get_wtime();
    
    // most imp line 
    #pragma opm parallel for reduction(min: min_par) reduction(max: max_par) reduction(+: sum_par)
    
    for(int i =0;i<n; i++){
        if(nums[i] < min_par) min_par = nums[i];
        if(nums[i] > max_par) max_par = nums[i];
        sum_par += nums[i];
    }
    
    end = omp_get_wtime();
    
    avg_par= (double)sum_par/n;
    double time_par = end - start;
    
    cout<<"--------Sequential Computation------"<<endl;
    cout<<"1.Minimum value : "<<min_seq<<endl;
    cout<<"2.Maximum value : "<<max_seq<<endl;
    cout<<"3.Sum value : "<<sum_seq<<endl;
    cout<<"4.Average value : "<<avg_seq<<endl;
    cout<<"5.Time value : "<<time_seq<<endl;
    cout<<"------------------------------------"<<endl;
    
    cout<<"--------Parallel Computation------"<<endl;
    cout<<"1.Minimum value : "<<min_par<<endl;
    cout<<"2.Maximum value : "<<max_par<<endl;
    cout<<"3.Sum value : "<<sum_par<<endl;
    cout<<"4.Average value : "<<avg_par<<endl;
    cout<<"5.Time value : "<<time_par<<endl;
    cout<<"------------------------------------"<<endl;
    
    cout<<"--------Time Difference --------"<<endl;
    cout<<"TD: "<<(time_seq / time_par)<<" x "<<endl;
    
    
    
    return 0;
}