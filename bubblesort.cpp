#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>

using namespace std;

// --------------------------------------------------
// Utility: Print Array (for debugging)
// --------------------------------------------------
void printArray(const vector<int>& arr) {
    for (int i = 0; i < arr.size(); i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

// --------------------------------------------------
// SEQUENTIAL BUBBLE SORT
// Time Complexity: O(n^2)
// --------------------------------------------------
void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// --------------------------------------------------
// PARALLEL BUBBLE SORT (Optimized)
// Using Odd-Even Transposition Sort
//
// Key Idea:
// - Even phase: (0,1), (2,3), ...
// - Odd phase:  (1,2), (3,4), ...
// - Each phase can run in parallel safely
//
// Optimization:
// - Single parallel region (avoids repeated thread creation)
// --------------------------------------------------
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();

    #pragma omp parallel
    {
        for (int i = 0; i < n; i++) {

            // Parallel loop for each phase
            #pragma omp for schedule(static)
            for (int j = i % 2; j < n - 1; j += 2) {
                if (arr[j] > arr[j + 1]) {
                    swap(arr[j], arr[j + 1]);
                }
            }

            // Implicit barrier ensures correctness
        }
    }
}

// --------------------------------------------------
// MAIN FUNCTION
// --------------------------------------------------
int main() {

    int n = 10000;   // Try smaller (5000) if too slow
    vector<int> arr(n);

    // Generate random array
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000;
    }

    // Copies for fair comparison
    vector<int> arr1 = arr;
    vector<int> arr2 = arr;

    cout << "Array Size: " << n << endl;

    double start, end;

    // -------------------------------
    // Sequential Bubble Sort
    // -------------------------------
    start = omp_get_wtime();
    sequentialBubbleSort(arr1);
    end = omp_get_wtime();
    cout << "\nSequential Bubble Sort Time: " << (end - start) << " sec";

    // -------------------------------
    // Parallel Bubble Sort
    // -------------------------------
    start = omp_get_wtime();
    parallelBubbleSort(arr2);
    end = omp_get_wtime();
    cout << "\nParallel Bubble Sort Time:   " << (end - start) << " sec";

    cout << endl;

    return 0;
}
