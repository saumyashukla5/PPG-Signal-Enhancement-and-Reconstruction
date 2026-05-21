#include <iostream>
#include <map>
#include <set>
#include <cstring>
#include <cctype>
using namespace std;

int n;
char production[10][10];
map<char,set<char>> first, follow;

void find_first(char c)
{
    if(!isupper(c))
    {
        first[c].insert(c);
        return;
    }

    for(int i=0;i<n;i++)
    {
        if(production[i][0]==c)
        {
            if(production[i][2]=='#')
                first[c].insert('#');

            else if(!isupper(production[i][2]))
                first[c].insert(production[i][2]);

            else
            {
                find_first(production[i][2]);
                for(char x:first[production[i][2]])
                    first[c].insert(x);
            }
        }
    }
}

void find_follow(char c)
{
    if(production[0][0]==c)
        follow[c].insert('$');

    for(int i=0;i<n;i++)
    {
        for(int j=2;j<strlen(production[i]);j++)
        {
            if(production[i][j]==c)
            {
                if(production[i][j+1]!='\0')
                {
                    char next=production[i][j+1];

                    if(!isupper(next))
                        follow[c].insert(next);
                    else
                    {
                        for(char x:first[next])
                        {
                            if(x!='#')
                                follow[c].insert(x);
                        }
                    }
                }

                else if(c!=production[i][0])
                {
                    find_follow(production[i][0]);
                    for(char x:follow[production[i][0]])
                        follow[c].insert(x);
                }
            }
        }
    }
}

int main()
{
    cout<<"Enter number of productions: ";
    cin>>n;

    cout<<"Enter productions:\n";

    for(int i=0;i<n;i++)
        cin>>production[i];

    for(int i=0;i<n;i++)
        find_first(production[i][0]);

    for(int i=0;i<n;i++)
        find_follow(production[i][0]);

    cout<<"\nFIRST sets\n";

    for(auto it:first)
    {
        cout<<"FIRST("<<it.first<<") = { ";
        for(char x:it.second)
            cout<<x<<" ";
        cout<<"}\n";
    }

    cout<<"\nFOLLOW sets\n";

    for(auto it:follow)
    {
        cout<<"FOLLOW("<<it.first<<") = { ";
        for(char x:it.second)
            cout<<x<<" ";
        cout<<"}\n";
    }

    return 0;
}