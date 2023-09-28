// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <iostream>
#include <vector>
using namespace std;

int main()
{ 
    int N, K;
    scanf("%d %d", &N, &K);

    vector<vector<int>> node_dict;
    node_dict.resize(N + 1);
    for (int i = 0; i <= N; i++)
    {
        node_dict[i].resize(0);
    }

    vector<vector<int>> distance;
    distance.resize(N + 1);
    for (int i = 0; i <= N; i++)
    {
        distance[i].resize(N+1, -1);
    }

    for (int i = 0; i < N - 1; i++)
    {
        int A, B, L;
        scanf("%d %d %d", &A, &B, &L);
        distance[A][B] = L;
        distance[B][A] = L;
        node_dict[A].push_back(B);
        node_dict[B].push_back(A);
    }
    printf("%d, %d", N, K);

    return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单


