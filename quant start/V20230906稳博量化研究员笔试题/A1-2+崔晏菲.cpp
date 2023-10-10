// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <iostream>
#include <vector>
using namespace std;

#define UNFINDED -1

vector<vector<int>> node_dict;
vector<vector<int>> dist;

int print_dist(vector<vector<int>> map)
{
    printf("--------------------------------\n");
    for (int i = 1; i < map.size(); i++)
    {
        for (int j = 1; j < map[i].size(); j++)
        {
            printf("%-3d ", map[i][j]);
        }
        printf("\n");
    }
    printf("--------------------------------\n");
    return 0;
}

int get_distance(int source, int A, int B)
{
    if (dist[A][B] != UNFINDED)
    {
        //printf("A:%d B:%d AB:%d\n", A, B, dist[A][B]);
        //print_dist(dist);
        return dist[A][B];
        
    }
    else
    {
        for(int j = 0; j<node_dict[A].size(); j++)
        {
            int C = node_dict[A][j];
            if(C != source)
            {
                dist[source][C] = dist[source][A] + dist[A][C];
                int CB = get_distance(A, C, B);
                if(CB != UNFINDED)
                {
                    int AB = CB + dist[A][C];
                    dist[A][B] = AB;
                    dist[B][A] = AB;
                    //printf("A:%d C:%d B:%d AB:%d CB:%d\n", A, C, B, AB, CB);
                    //print_dist(dist);
                    return AB;
                }
            }
        }
        return UNFINDED;
    }
}

int main()
{
    FILE *f;
    f = fopen("./output.txt", "w");
    vector<int> ANS;
    ANS.resize(0);

    while(true)
    {
        int N, K;
        scanf("%d %d", &N, &K);
        if(N == 0 && K == 0)
        {
            break;
        }

        dist.resize(N + 1);
        for (int i = 0; i <= N; i++)
        {
            dist[i].resize(N + 1, UNFINDED);
        }

        node_dict.resize(N + 1);
        for (int i = 0; i <= N; i++)
        {
            node_dict[i].resize(0);
        }

        for (int i = 0; i < N - 1; i++)
        {
            int A, B, L;
            scanf("%d %d %d", &A, &B, &L);
            dist[A][B] = L;
            dist[B][A] = L;
            node_dict[A].push_back(B);
            node_dict[B].push_back(A);
        }

        int ans = 0;
        for(int i = 1; i <= N; i++)
        {
            for(int j = 1; j <= N; j++)
            {
                if (i != j)
                {
                    int dis = get_distance(0, i, j);
                    if (dis <= K && dis > 0)
                    {
                        ans++;
                    }
                }
            }
        }
        ans /= 2;
        ANS.push_back(ans);
    }

    for(int i = 0; i<ANS.size(); i++)
    {
        printf("%d\n", ANS[i]);
        fprintf(f, "%d\n", ANS[i]);
    }

    return 0;
}