#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// -------------------------------
// MERGE FUNCTION
// -------------------------------
void merge(vector<int>& arr, int st, int mid, int ed) {
    int n1 = mid - st + 1;
    int n2 = ed - mid;

    vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; i++) L[i] = arr[st + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = st;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

// -------------------------------
// SEQUENTIAL MERGE SORT
// -------------------------------
void sequentialMergeSort(vector<int>& arr, int st, int ed) {
    if (st >= ed) return;

    int mid = (st + ed) / 2;

    sequentialMergeSort(arr, st, mid);
    sequentialMergeSort(arr, mid + 1, ed);

    merge(arr, st, mid, ed);
}

// -------------------------------
// PARALLEL MERGE SORT
// -------------------------------
void parallelMergeSort(vector<int>& arr, int st, int ed) {
    if (st >= ed) return;

    int mid = (st + ed) / 2;

    #pragma omp parallel sections
    {
        #pragma omp section
        parallelMergeSort(arr, st, mid);

        #pragma omp section
        parallelMergeSort(arr, mid + 1, ed);
    }

    merge(arr, st, mid, ed);
}

// -------------------------------
// MAIN FUNCTION
// -------------------------------
int main() {

    int n = 10000;
    vector<int> arr(n);

    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000;
    }

    vector<int> arr1 = arr;
    vector<int> arr2 = arr;

    cout << "Array Size: " << n << endl;

    double start, end;

    // Sequential
    start = omp_get_wtime();
    sequentialMergeSort(arr1, 0, n - 1);
    end = omp_get_wtime();
    cout << "\nSequential Merge Sort Time: " << (end - start) << " sec";

    // Parallel
    start = omp_get_wtime();
    parallelMergeSort(arr2, 0, n - 1);
    end = omp_get_wtime();
    cout << "\nParallel Merge Sort Time:   " << (end - start) << " sec";

    cout << endl;

    return 0;
}

// 1. Using GCC / g++ (Linux, WSL, MinGW, etc.)
// ✅ Compile
// g++ -fopenmp your_file.cpp -o bubble
// ▶️ Run
// ./bubble


// 🪟 2. On Windows (MinGW / MSYS2)
// Make sure your g++ supports OpenMP.
// Compile:
// g++ -fopenmp your_file.cpp -o bubble.exe
// Run:
// bubble.exe
