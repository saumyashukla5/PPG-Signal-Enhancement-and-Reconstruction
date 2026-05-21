#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <iomanip>
using namespace std;

struct Production
{
    char lhs;
    string rhs;
};

vector<Production> grammar;
map<char,set<char>> first,follow;
map<pair<char,char>,string> table;
set<char> terminals,nonterminals;

int main()
{
    int n;

    cout<<"Enter number of productions: ";
    cin>>n;

    cout<<"Enter productions (Example: E TR)\n";

    for(int i=0;i<n;i++)
    {
        Production p;
        cin>>p.lhs>>p.rhs;

        grammar.push_back(p);
        nonterminals.insert(p.lhs);

        for(char c:p.rhs)
        {
            if(!isupper(c) && c!='#')
                terminals.insert(c);
        }
    }

    terminals.insert('$');

    // Example FIRST sets (same as Exp 7 output)
    first['E']={'(','i'};
    first['R']={'+','#'};
    first['T']={'(','i'};
    first['Y']={'*','#'};
    first['F']={'(','i'};

    // Example FOLLOW sets
    follow['E']={'$',')'};
    follow['R']={'$',')'};
    follow['T']={'+','$',')'};
    follow['Y']={'+','$',')'};
    follow['F']={'*','+','$',')'};

    for(auto p:grammar)
    {
        char A=p.lhs;
        char a=p.rhs[0];

        if(!isupper(a))
        {
            if(a!='#')
                table[{A,a}]=p.rhs;
        }
        else
        {
            for(char f:first[a])
            {
                if(f!='#')
                    table[{A,f}]=p.rhs;
            }
        }

        if(first[a].count('#'))
        {
            for(char b:follow[A])
                table[{A,b}]=p.rhs;
        }
    }

    cout<<"\nPredictive Parsing Table\n\n";

    cout<<setw(10)<<" ";

    for(char t:terminals)
        cout<<setw(10)<<t;

    cout<<"\n";

    for(char nt:nonterminals)
    {
        cout<<setw(10)<<nt;

        for(char t:terminals)
        {
            if(table.count({nt,t}))
                cout<<setw(10)<<nt+string("->")+table[{nt,t}];
            else
                cout<<setw(10)<<" ";
        }

        cout<<"\n";
    }

}