#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define THRESHOLD 1000

void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

void mergeSortSequential(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortSequential(arr, left, mid);
        mergeSortSequential(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void mergeSortParallel(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        if ((right - left) <= THRESHOLD) {
            mergeSortSequential(arr, left, mid);
            mergeSortSequential(arr, mid + 1, right);
        } else {
            #pragma omp parallel sections
            {
                #pragma omp section
                mergeSortParallel(arr, left, mid);

                #pragma omp section
                mergeSortParallel(arr, mid + 1, right);
            }
        }
        merge(arr, left, mid, right);
    }
}

void bubbleSortOddEvenParallel(int arr[], int n) {
    int isSorted = 0;
    while (!isSorted) {
        isSorted = 1;

        #pragma omp parallel for
        for (int i = 1; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                isSorted = 0;
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                isSorted = 0;
            }
        }
    }
}

void bubbleSortOddEvenSequential(int arr[], int n) {
    int isSorted = 0;
    while (!isSorted) {
        isSorted = 1;

        for (int i = 1; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                isSorted = 0;
            }
        }

        for (int i = 0; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                isSorted = 0;
            }
        }
    }
}

void generateRandomArray(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }
}

int main() {
    srand(time(0));
    omp_set_num_threads(omp_get_max_threads());  // Use all available threads

    int n;
    printf("Enter the size of the array: ");
    scanf("%d", &n);

    int *arr1 = (int *)malloc(n * sizeof(int));
    int *arr2 = (int *)malloc(n * sizeof(int));
    int *arr3 = (int *)malloc(n * sizeof(int));
    int *arr4 = (int *)malloc(n * sizeof(int));

    printf("\nGayatri Kurulkar 41039 BE A\n");
    printf("+-----------+------------------------------+------------------------------+----------------------+----------------------+------------+-------------+\n");
    printf("| Iteration | Odd-Even Bubble Sort Seq Time | Odd-Even Bubble Sort Par Time | Merge Sort Seq Time | Merge Sort Par Time | Speedup    | Efficiency  |\n");
    printf("+-----------+------------------------------+------------------------------+----------------------+----------------------+------------+-------------+\n");

    for (int i = 0; i < 5; i++) {
        generateRandomArray(arr1, n);
        for (int j = 0; j < n; j++) {
            arr2[j] = arr1[j];
            arr3[j] = arr1[j];
            arr4[j] = arr1[j];
        }

        double start_time, end_time;

        // Odd-Even Bubble Sort Sequential
        start_time = omp_get_wtime();
        bubbleSortOddEvenSequential(arr1, n);
        end_time = omp_get_wtime();
        double bubbleSeqTime = end_time - start_time;

        // Odd-Even Bubble Sort Parallel
        start_time = omp_get_wtime();
        bubbleSortOddEvenParallel(arr2, n);
        end_time = omp_get_wtime();
        double bubbleParTime = end_time - start_time;

        // Merge Sort Sequential or Insertion
        start_time = omp_get_wtime();
        if (n < 20) insertionSort(arr3, n);
        else mergeSortSequential(arr3, 0, n - 1);
        end_time = omp_get_wtime();
        double mergeSeqTime = end_time - start_time;

        // Merge Sort Parallel or Insertion
        start_time = omp_get_wtime();
        if (n < 20) insertionSort(arr4, n);
        else mergeSortParallel(arr4, 0, n - 1);
        end_time = omp_get_wtime();
        double mergeParTime = end_time - start_time;

        double bubbleSpeedup = bubbleSeqTime / bubbleParTime;
        double mergeSpeedup = mergeSeqTime / mergeParTime;
        double efficiency = (bubbleSpeedup + mergeSpeedup) / (2 * omp_get_max_threads());

        printf("| %9d | %28f | %28f | %20f | %20f | %10f | %11f |\n",
               i + 1, bubbleSeqTime, bubbleParTime, mergeSeqTime, mergeParTime,
               bubbleSpeedup + mergeSpeedup, efficiency);
    }

    printf("+-----------+------------------------------+------------------------------+----------------------+----------------------+------------+-------------+\n");

    free(arr1);
    free(arr2);
    free(arr3);
    free(arr4);
    return 0;
}
