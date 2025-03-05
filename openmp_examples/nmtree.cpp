#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <omp.h>


// to avoid generate duplicated tree, give rule as below:
// 
// 1. The longest path fo the tree put in the leftest, it means if do a in-order DFS, the longest path leaf would be the first leaf to visit. 
        // Let's call this longest path as trunk path, the trunk path length we can call it trunk weight(trunk nodes), trank weight is the nodes number in the trunk path. 
        // let leaf node height to be 1, and every node have a height.
// 2. All other pathes, up to M - 1, we call these branch pathes, branch pathes we can regard them be sorted from left to right, the weight of branch we count the edges from the fork node.
// 3. As we already define the longest path is the trunk path, so the depth of branch path can't more than trunk path, 
      //the branch path depth == (depth of trunk node from) + (branch weight), we can use height to check.
// 4. we can classify the tree as a M element vector (trunk weight, weight, weight, ... ), weight can at max be trunk weight - 1, and aslo care the fork node depth add weight can't more than trunk depth
// 5. when hook up next branch, choose the rightest path to hook, this way to keep order, avoid duplicated 
// eg  for the N=7, M=2 tree, we can have (7) (6,1){ the branch can from different fork node}, (5,2) (4,3)
//     for the N=7, M=3 tree  we can have (7)(6,1)(5,2)(5,1,1)(4,3)(4,1,2)(4,2,1),(3,2,2) each one means a type, in this type may have different trees.
     // note  two branches if short one is left, then two branch can't hook on same fork node to avoid duplicate, eg (4, 2,1) branch 2 and 1 can hook on same node, but (4, 1, 2) branch 1 and 2 can't
// 6. In this algorithm, first we get all types, it is a vector has M elements, and then base on this type, build the trees for this type.
// 7. There still be a small chance to be duplicated, then dump all tree to a topo sorted string, to deduplicated.
// 8. when print out, the lefter path will printed upperside, the righter path will printed downside, the print out is a folder tree like.
// 9. When build the trees based on type, it can do parallel computing.

using namespace std;


class TreeNode {
public:
    std::vector<TreeNode*> children;
    int height;
    TreeNode() = default;
    TreeNode(int h){
        height = h;
    }
};

unordered_map<string,int> treestrmap;
int cnt_trees = 0;
int cnt_unitrees = 0;

void printConnections(TreeNode* node, const std::string& prefix = "") {
    if (!node) return;

    std::cout << prefix << node->height << std::endl;

    for (size_t i = 0; i < node->children.size(); ++i) {
        std::string newPrefix = prefix + (i == node->children.size() - 1 ? "    " : "|   ");
        printConnections(node->children[i], newPrefix + "|-- ");
    }
}

void getpermu(int nr, int mr, vector<int> & cur, vector<vector<int>> & ret ){
    if(nr == 0) {
        ret.push_back(cur);
        return;
    };
    if(mr == 1){
        if(nr < cur[0]){
            cur.push_back(nr);
            ret.push_back(cur);
            cur.pop_back();
        }
        return;
    } else {
        for(int i = min(nr, cur[0] - 1) ; i > 0; --i){
            cur.push_back(i);
            getpermu(nr - i, mr - 1, cur, ret);
            cur.pop_back();
        }
    }
}

vector<vector<int>> gen_vectors(int n, int m){
    vector<vector<int>> ret;
    for(int trunk = n; trunk > n/m; --trunk){
        vector<int> cur;
        cur.push_back(trunk);
        getpermu(n - trunk, m - 1, cur, ret);
    }
    return ret;
}

void print_vec(const vector<vector<int>> & vects){
    for(const auto & vec : vects){
        cout << "(";
        for(auto & v : vec){
            cout << " " << v;
        }
        cout << ")" << endl;
    }
}

TreeNode * getbranch(int len){
    TreeNode * root = new TreeNode(len);
    auto cur = root;
    --len;
    while(len){
        cur->children.push_back(new TreeNode(len));
        cur = cur->children[0];
        --len;
    }
    return root;
}

void deletebranch(TreeNode * root){
    while(root){
        if(root->children.size() > 0){
            auto next = root->children[0];
            delete root;
            root = next;
        } else {
            delete root;
            root = nullptr;
        }
    }
}

void printbranch(TreeNode * branch){
    while(branch){
        cout << " " << branch->height ;
        if(branch->children.size() > 0){
            branch = branch->children[0];
        } else {
            break;
        }
    }
    cout << endl;
}

int get_w_and_h(TreeNode * root, int & h){
    if(!root) {
        h = 0;
        return 0;
    }
    if(root->children.size() == 0){
        h = 1;
        return 1;
    }
    int sumw = 1;
    for(auto & child : root->children){
        int ch = 0;
        sumw += get_w_and_h(child, ch);
        h = max(h, ch + 1);
    }
    return sumw;
}

void savetostring(TreeNode* root, string & ret){
    if(!root) return;
    ret += to_string(root->height);
    //ret += "--";
    vector<TreeNode*> cpy = root->children;
    // sort by weight and height
    sort(cpy.begin(), cpy.end(), [&](TreeNode * a , TreeNode * b)->bool{
        int hca = 0;
        int hcb = 0;
        int wca = get_w_and_h(a, hca);
        int wcb =get_w_and_h(b, hcb);
        if(wca > wcb) return true;
        else if(wca < wcb) return false;
        else {
            if(hca > hcb) return true;
            else if (hca < hcb) return false;
            else return true;
        }
        
    });
    for(auto & child : cpy){
        ret += "(";
        savetostring(child, ret);
        ret += ")";
    }
}
// the lefter path will printed upperside, the righter path will printed downside
void print_tree(TreeNode * root, const vector<int> & vec){
    string str;
    savetostring(root, str);
    bool nondup = false;
    #pragma omp critical
    {
        auto it = treestrmap.find(str);
        if(it == treestrmap.end()) {
            nondup = true;
        } else {
            nondup = false;
        }
        treestrmap[str] += 1;
    }
    if(nondup) {
        cnt_unitrees += 1;
        cout << "\nPrint tree====================== \nVector: (";
        for(auto & v : vec) cout << v << " ";
        cout << ")" << endl; 
        cout << "Topo string: " << str << endl;
        printConnections(root);
    }
}

void build_tree(int idx, TreeNode * root, const vector<int> & vec, vector<TreeNode *> & subtrees){
    if(idx == subtrees.size()) {
        print_tree(root, vec);
        ++cnt_trees;
        return;
    }
    TreeNode * hook = root;
    while(hook->children.size() != 0){
        // check the branch to be hook must be the shortest branch of hook node
        bool can_hook = true;
        for(auto & branch : hook->children){
            if(vec[idx] > branch->height) {
                can_hook = false;
                break;
            }
        }
        if(!can_hook) break;
        // hook
        hook->children.push_back(subtrees[idx]);
        // dfs
        build_tree(idx + 1, root, vec, subtrees);
        // unhook
        hook->children.pop_back();
        hook = hook->children.back();
    }

}

void generateTree(int n, int m, const vector<int> & vec){
    vector<TreeNode *> subtrees;
    for(int i = 0; i < vec.size(); ++i){
        subtrees.push_back(getbranch(vec[i]));
    }
    TreeNode * root = subtrees[0];
    build_tree(1, root, vec, subtrees);
    for(auto & branch : subtrees){
        deletebranch(branch);
    }
}

int main(int argc, char*argv[]) {
    // get N,M from command line
    if(argc < 3){
        cout << "Usage: N and M parameter are expected, eg ./nmtree 8 5" << endl;
        return 1;
    }

    int N = stoi(argv[1]);
    int M = stoi(argv[2]);
    cout << "N = " <<  N << ", M = " << M << endl;
    auto vects = gen_vectors(N, M);
    cout << "Vectors: " << endl;
    print_vec(vects);

    #pragma omp parallel
    {
        #pragma omp for
        for(auto & vec : vects){
            generateTree(N, M, vec);
        }
    }

    cout << "Unique topo trees: " << cnt_unitrees << " trees " << endl;

    return 0;
}