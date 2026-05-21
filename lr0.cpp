#include <bits/stdc++.h>
using namespace std;

struct Item {
    string lhs, rhs;
    int dot;
};

vector<pair<string, string>> grammar;

// check if item exists
bool exists(vector<Item> &I, Item x) {
    for (auto i : I) {
        if (i.lhs == x.lhs && i.rhs == x.rhs && i.dot == x.dot)
            return true;
    }
    return false;
}

// closure function
vector<Item> closure(vector<Item> I) {
    bool added = true;

    while (added) {
        added = false;

        for (int i = 0; i < I.size(); i++) {
            if (I[i].dot < I[i].rhs.size()) {
                char sym = I[i].rhs[I[i].dot];

                if (isupper(sym)) {
                    for (auto p : grammar) {
                        if (p.first[0] == sym) {
                            Item newItem = {p.first, p.second, 0};

                            if (!exists(I, newItem)) {
                                I.push_back(newItem);
                                added = true;
                            }
                        }
                    }
                }
            }
        }
    }
    return I;
}

// goto function
vector<Item> GOTO(vector<Item> I, char X) {
    vector<Item> temp;

    for (auto i : I) {
        if (i.dot < i.rhs.size() && i.rhs[i.dot] == X) {
            temp.push_back({i.lhs, i.rhs, i.dot + 1});
        }
    }

    return closure(temp);
}

// compare states
bool same(vector<Item> &a, vector<Item> &b) {
    if (a.size() != b.size()) return false;

    for (auto i : a) {
        if (!exists(b, i)) return false;
    }
    return true;
}

// print state
void printState(vector<Item> &I, int id) {
    cout << "\nI" << id << ":\n";
    for (auto i : I) {
        cout << i.lhs << " -> ";
        string s = i.rhs;
        s.insert(i.dot, ".");
        cout << s << endl;
    }
}

int main() {
    int n;
    cout << "Enter number of productions: ";
    cin >> n;

    cout << "Enter productions (E->E+T):\n";
    for (int i = 0; i < n; i++) {
        string s;
        cin >> s;
        int pos = s.find("->");
        grammar.push_back({s.substr(0, pos), s.substr(pos + 2)});
    }

    // augment grammar
    string start = grammar[0].first;
    grammar.insert(grammar.begin(), {start + "'", start});

    vector<vector<Item>> states;

    // initial state
    vector<Item> I0 = closure({{start + "'", start, 0}});
    states.push_back(I0);

    queue<int> q;
    q.push(0);

    while (!q.empty()) {
        int i = q.front();
        q.pop();

        set<char> symbols;

        for (auto item : states[i]) {
            if (item.dot < item.rhs.size()) {
                symbols.insert(item.rhs[item.dot]);
            }
        }

        for (char X : symbols) {
            vector<Item> next = GOTO(states[i], X);
            if (next.empty()) continue;

            bool found = false;
            for (int j = 0; j < states.size(); j++) {
                if (same(states[j], next)) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                states.push_back(next);
                q.push(states.size() - 1);
            }
        }
    }

    // print all states
    for (int i = 0; i < states.size(); i++) {
        printState(states[i], i);
    }

    return 0;
}