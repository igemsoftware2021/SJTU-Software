#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <sys/timeb.h>
#include <ctype.h>
#include <termios.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/select.h>

#define SCREENWIDTH 90
#define FILENAME_LIMIT 80
#define MAX_DOMAIN_LENGTH 100
#define NB_ENABLE 1
#define NB_DISABLE 0
#define MAX_MUTATIONS 10 // maximum number of simultaneous mutations
// Domain最长60个碱基, 最大发生10次mutations(?),剩余没看懂

double GCstr = 2;
double ATstr = 1;
double GTstr = 0;
double MBstr = -3; // mismatch, bulge
double LLstr = -0.5; // large loop?   文章说的是每多一个碱基有一个-0.5
double DHstr = 3; // score for domain ending in a base pair  解决interface_problem
//用来计算crosstalk与interaction的得分

int MAX_IMPORTANCE = 100; //不同domain重要性

int LHbases = 4;
double LHstart = 2;
double LHpower = 2;        //以上应该是用于处理连续配对的
double INTRA_SCORE = 5; // score bonus for intrastrand/dimerization interactions
double CROSSTALK_SCORE = -5; // score bonus for crosstalk (as compared to interaction)  //文中说的是-10?
double CROSSTALK_DIV = 2; // crosstalk score is divided by this much (and then score is subtracted)


double GGGG_PENALTY = 50;
double ATATAT_PENALTY = 20;  //两种特殊的罚分 intrinsic_score就只有这两项
// ****************************以上是罚分部分******************************************

double pairscore(int* seq1, int len1, int* seq2, int len2) {
    // Gives the score of the two sequences's crosstalk
    double score, temp;
    int i, j, k;
    double** Cmatrix; // complementarity matrix
    double** Smatrix; // score matrix
    int** SDmatrix; // running total of helix size, 0 if current base didn't contribute.

    // Memory allocations
    if ((Cmatrix = malloc(len1 * sizeof(double*))) == NULL) {
        fprintf(stderr, "Insufficient memory for score calculations!\n");
        exit(-1);
    }
    if ((Smatrix = malloc(len1 * sizeof(double*))) == NULL) {
        fprintf(stderr, "Insufficient memory for score calculations!\n");
        exit(-1);
    }
    if ((SDmatrix = malloc(len1 * sizeof(int*))) == NULL) {
        fprintf(stderr, "Insufficient memory for score calculations!\n");
        exit(-1);
    }

    for (i = 0; i < len1; i++) {
        if ((Cmatrix[i] = malloc(len2 * sizeof(double))) == NULL) {
            fprintf(stderr, "Insufficient memory for score calculations!\n");
            exit(-1);
        }
        if ((Smatrix[i] = malloc(len2 * sizeof(double))) == NULL) {
            fprintf(stderr, "Insufficient memory for score calculations!\n");
            exit(-1);
        }
        if ((SDmatrix[i] = malloc(len2 * sizeof(int))) == NULL) {
            fprintf(stderr, "Insufficient memory for score calculations!\n");
            exit(-1);
        }
    }

    // 动态规划来优化?
    // Seed complementarity matrix
    for (i = 0; i < len1; i++) {
        for (j = 0; j < len2; j++) {
            if (((seq1[i] + seq2[len2-1-j])%10 == 5)&&((seq1[i] * seq2[len2-1-j])%10 == 4)) // G/C Match
                Cmatrix[i][j] = GCstr;
            else if (((seq1[i] + seq2[len2-1-j])%10 == 5)&&((seq1[i] * seq2[len2-1-j])%10 == 6)) // A/T Match
                Cmatrix[i][j] = ATstr;
            else if (((seq1[i] + seq2[len2-1-j])%10 == 4)&&((seq1[i] * seq2[len2-1-j])%10 == 3)) // G/T Wobble
                Cmatrix[i][j] = GTstr;
            else
                Cmatrix[i][j] = MBstr; // mismatch
        }
    }

    // Calculate score matrix
    score = 0;

    Smatrix[0][0] = Cmatrix[0][0];
    if (Smatrix[0][0] < 0) {
        Smatrix[0][0] = 0;
        SDmatrix[0][0] = 0;
    } else {
        Smatrix[0][0] = Smatrix[0][0] + DHstr;
        SDmatrix[0][0] = 1;
    }

    if (Smatrix[0][0] > score)
        score = Smatrix[0][0];

    for (j = 1; j < len2; j++) {
        Smatrix[0][j] = Cmatrix[0][j];
        if (Smatrix[0][j] < 0) {
            Smatrix[0][j] = 0;
            SDmatrix[0][j] = 0;
        } else {
            Smatrix[0][j] = Smatrix[0][j] + DHstr;
            SDmatrix[0][j] = 1;
        }
        if (Smatrix[0][j] > score)
            score = Smatrix[0][j];
    }


    for (i = 1; i < len1; i++) {
        Smatrix[i][0] = Cmatrix[i][0];
        if (Smatrix[i][0] < 0) {
            Smatrix[i][0] = 0;
            SDmatrix[i][0] = 0;
        } else {
            Smatrix[i][0] = Smatrix[i][0] + DHstr;
            SDmatrix[i][0] = 1;
        }
        if (Smatrix[i][0] > score)
            score = Smatrix[i][0];

        for (j = 1; j < len2; j++) {

            if (Cmatrix[i][j] < 0) { // destabilizing base
                SDmatrix[i][j] = 0;
                Smatrix[i][j] = 0;

                if ((SDmatrix[i-1][j-1] > 0)&&(Smatrix[i-1][j-1] + MBstr > 0)) // starting a mismatch loop
                    Smatrix[i][j] = Smatrix[i-1][j-1] + MBstr;
                if ((SDmatrix[i-1][j-1] == 0)&&(Smatrix[i-1][j-1] + LLstr > 0)) // expanding a mismatch loop
                    Smatrix[i][j] = Smatrix[i-1][j-1] + LLstr;

                if ((SDmatrix[i][j-1] > 0)&&(Smatrix[i][j-1] + MBstr > 0)&&(Smatrix[i][j-1] + MBstr > Smatrix[i][j]))
                    Smatrix[i][j] = Smatrix[i][j-1] + MBstr;
                if ((SDmatrix[i][j-1] == 0)&&(Smatrix[i][j-1] + LLstr > 0)&&(Smatrix[i][j-1] + LLstr > Smatrix[i][j]))
                    Smatrix[i][j] = Smatrix[i][j-1] + LLstr;

                if ((SDmatrix[i-1][j] > 0)&&(Smatrix[i-1][j] + MBstr > 0)&&(Smatrix[i-1][j] + MBstr > Smatrix[i][j]))
                    Smatrix[i][j] = Smatrix[i-1][j] + MBstr;
                if ((SDmatrix[i-1][j] == 0)&&(Smatrix[i-1][j] + LLstr > 0)&&(Smatrix[i-1][j] + LLstr > Smatrix[i][j]))
                    Smatrix[i][j] = Smatrix[i-1][j] + LLstr;

                if (Smatrix[i][j] < 0)
                    Smatrix[i][j] = 0;

            } else { // stabilizing base
                Smatrix[i][j] = Cmatrix[i][j];
                SDmatrix[i][j] = 1;

                if ((SDmatrix[i-1][j-1] > 0)&&(Smatrix[i-1][j-1] > 0)) { // continuing a helix
                    Smatrix[i][j] = Smatrix[i-1][j-1] + Cmatrix[i][j];
                    SDmatrix[i][j] = SDmatrix[i-1][j-1] + 1;
                } else if ((SDmatrix[i-1][j-1] == 0)&&(Smatrix[i-1][j-1] > 0)) { // starting a new helix
                    Smatrix[i][j] = Smatrix[i-1][j-1] + Cmatrix[i][j];
                    SDmatrix[i][j] = 1;
                }

                if ((SDmatrix[i][j-1] > 0)&&(Smatrix[i][j-1] > 0)&&(Smatrix[i][j-1] + Cmatrix[i][j] - Cmatrix[i][j-1] + MBstr > Smatrix[i][j])) {
                    Smatrix[i][j] = Smatrix[i][j-1] + Cmatrix[i][j] - Cmatrix[i][j-1] + MBstr; // introducing a 1-bulge, destroying previous bond
                    SDmatrix[i][j] = 1;
                } else if ((SDmatrix[i][j-1] == 0)&&(Smatrix[i][j-1] > 0)&&(Smatrix[i][j-1] + Cmatrix[i][j] > Smatrix[i][j])) {
                    Smatrix[i][j] = Smatrix[i][j-1] + Cmatrix[i][j]; // closing a bulge
                    SDmatrix[i][j] = 1;
                }

                if ((SDmatrix[i-1][j] > 0)&&(Smatrix[i-1][j] > 0)&&(Smatrix[i-1][j] + Cmatrix[i][j] - Cmatrix[i-1][j] + MBstr > Smatrix[i][j])) {
                    Smatrix[i][j] = Smatrix[i-1][j] + Cmatrix[i][j] - Cmatrix[i-1][j] + MBstr;
                    SDmatrix[i][j] = 1;
                } else if ((SDmatrix[i-1][j] == 0)&&(Smatrix[i-1][j] > 0)&&(Smatrix[i-1][j] + Cmatrix[i][j] > Smatrix[i][j])) {
                    Smatrix[i][j] = Smatrix[i-1][j] + Cmatrix[i][j];
                    SDmatrix[i][j] = 1;
                }

                if (SDmatrix[i][j] > LHbases) {
                    // Extra points for long helices
                    temp = LHstart;
                    for (k = LHbases; k < SDmatrix[i][j]; k++)
                        temp = temp * LHpower;
                    Smatrix[i][j] = Smatrix[i][j] + temp;
                }
            }

            if ((SDmatrix[i][j] > 0)&&((i == (len1-1))||(j == (len2-1))))
                Smatrix[i][j] = Smatrix[i][j] + DHstr;

            if (Smatrix[i][j] > score)
                score = Smatrix[i][j];

        }
    }

    // Memory deallocation
    for (i = 0; i < len1; i++) {
        free(Cmatrix[i]);
        free(Smatrix[i]);
        free(SDmatrix[i]);
    }
    free(Cmatrix);
    free(Smatrix);
    free(SDmatrix);

    return score;
}

void DisplayBase(int base) {
    // 1 = G, 2 = A, 3 = T, 4 = C; 11 = G (locked), etc
    if (base == 1)
        printf("G");
    else if (base == 2)
        printf("A");
    else if (base == 3)
        printf("T");
    else if (base == 4)
        printf("C");
    else if (base == 11)
        printf("G");
    else if (base == 12)
        printf("A");
    else if (base == 13)
        printf("T");
    else if (base == 14)
        printf("C");
    else {
        fprintf(stderr, "Unknown base! %d \n", base);
        exit(-1);
    }
}

void format(char* seq, int len){
    int i;
    for(i=0;i<len;i++){
        if(seq[i]=='u'||seq[i]=='U') seq[i] = 'T';
        if(seq[i]>='a'&&seq[i]<='z') seq[i] = seq[i] - 'a' + 'A';
    }
}

int* char2int(char* seq,int len){
    int* seq_new= malloc(len*sizeof(int));
    int i;
    for(i=0;i<len;i++){
        switch(seq[i]){
            case 'G':seq_new[i]=1;break;
            case 'A':seq_new[i]=2;break;
            case 'T':seq_new[i]=3;break;
            case 'C':seq_new[i]=4;break;
        }
    }
    return seq_new;
}

int main(int argc,char** argv) {
    FILE* fp;
    int i, j, k, x;
    double score, old_score, old_d_intrinsic; // Score of system
    double* domain_score; // domain score
    int worst_domain; // domain that causes the worst score
    int num_mut;
    int mut_domain; // Domain, base, old, and new values
    int* mut_base;
    int* mut_new;
    int* mut_old;

    double** crosstalk; // Crosstalk is for domain similarity
    double** interaction; // Interaction is for domain complementarity
    double* domain_intrinsic; // intrinsic score to domains from various rules

    int doneflag, pausemode;
    char tempchar, tempchar2;
    double tempdouble;
    int num_domain, num_new_domain;
    char buffer[120];
    int temp_domain[MAX_DOMAIN_LENGTH];
    int* domain_length;
    int* domain_importance;
    int* domain_gatc_avail;
    int** domain; // 1 = G, 2 = A, 3 = T, 4 = C; 11 = G (locked), etc

    long num_mut_attempts, total_mutations;
    int rule_4g, rule_6at, rule_ccend, rule_ming, rule_init, rule_lockold, rule_targetworst, rule_gatc_avail;

    int dom1[30];
    int dom2[30];
    int len1, len2;

    struct timeb* curtime;

    // Randomize seed
    curtime = malloc(sizeof(struct timeb));
    ftime(curtime);
    srand(curtime->millitm);

    // Starting new design 开始设计!!************************************************************
    // 参数
    // num_domain: 链的条数
    // domain[]: num_domain*sizeof(int*) 存储每个domain  domain pointers??
    // domain_length[]: num_domain*sizeof(int)
    // domain_gatc_avail[]: num_domain*sizeof(int) 每个domain允许那些碱基
    // domain_importance[]: num_domain*sizeof(int) 每个domain的重要性

    // getfile
    char* filename = argv[1];
    fp = fopen(filename,"r");
    if( fp == NULL ){
        printf("Fail to open file!\n");
        exit(0);  //退出程序（结束程序）
    }

    // get num_domain
    char* tmp_str= malloc((MAX_DOMAIN_LENGTH+1)*sizeof(char));
    fgets(tmp_str,MAX_DOMAIN_LENGTH+1,fp);
    num_domain = atoi(tmp_str);
    // printf("%d\n",num_domain);

    // allocate memory1
    char** char_domain = malloc(num_domain*sizeof(char*));

    if ((domain = malloc(num_domain * sizeof(int*))) == NULL) {
        fprintf(stderr, "Insufficient memory for declaring domain pointers!\n");
        exit(-1);
    }

    if ((domain_length = malloc(num_domain * sizeof(int))) == NULL) {
        fprintf(stderr, "Insufficient memory for declaring domain lengths!\n");
        exit(-1);
    }

    if ((domain_gatc_avail = malloc(num_domain * sizeof(int))) == NULL) {
        fprintf(stderr, "Insufficient memory for declaring domain base availability!\n");
        exit(-1);
    }

    if ((domain_importance = malloc(num_domain * sizeof(int))) == NULL) {
        fprintf(stderr, "Insufficient memory for declaring domain importances!\n");
        exit(-1);
    }

    // get domain_length[] and char_domain[]
    int tmp_num = 0; //记录这是第几个domain
    while(fgets(tmp_str,MAX_DOMAIN_LENGTH+1,fp)!=NULL){
        if(tmp_str[0]=='>'){
            tmp_num++;
            fgets(tmp_str,MAX_IMPORTANCE+1,fp);
            domain_length[tmp_num-1] = strlen(tmp_str)-1;

            char_domain[tmp_num-1] = malloc((domain_length[tmp_num-1]+1)*sizeof(char));
            for(i=0;i<domain_length[tmp_num-1];i++) char_domain[tmp_num-1][i] = tmp_str[i];
            char_domain[tmp_num-1][i] = '\0';
        }
    }

    // allocate memory2
    for (i = 0; i < num_domain; i++) {
        while ((domain_length[i] <= 0)||(domain_length[i] > MAX_DOMAIN_LENGTH)) {
            printf("Domain lengths must be between 1 and %d!\n", MAX_DOMAIN_LENGTH);
            exit(-1);
        }
        if ((domain[i] = malloc(domain_length[i] * sizeof(int))) == NULL) {
            fprintf(stderr, "Insufficient memory for declaring domain bases!\n");
            exit(-1);
        }
        domain_importance[i] = 1;
    }

    // get domain[]
    for(i=0;i<num_domain;i++){
        format(char_domain[i],domain_length[i]);
        domain[i] = char2int(char_domain[i],domain_length[i]);
    }

    //调节模式与分数 启用哪些规则
    rule_4g = 1; // cannot have 4 G's or 4 C's in a row
    rule_6at = 1; // cannot have 6 A/T bases in a row
    rule_ccend = 1; // domains MUST start and end with C
    rule_ming = 1; // design tries to minimize usage of G
    rule_init = 7; // 1 = polyN, 2 = poly-H, 3 = poly-Y, 4 = poly-T
    rule_targetworst = 1; // target worst domains for mutation
    rule_gatc_avail = 15; // all flags set (bases available)
    doneflag = 0;

    for(i=0;i<num_domain;i++) domain_gatc_avail[i] = rule_gatc_avail;

    // Generate starting domain sequences 先保留下来 到时候可能会涉及到随机生成
    // 1 = G, 2 = A, 3 = T, 4 = C; 11 = G (locked), etc
    /*for (i = 0; i < num_domain; i++) {
        //domain_gatc_avail[] 在这里设置
        domain_gatc_avail[i] = rule_gatc_avail;
        //随机生成序列 对domain[i] 分配序列
        for (j = 0; j < domain_length[i]; j++) {
            domain[i][j] = 0;
            while (domain[i][j] == 0) {
                k = int_urn(1,4);
                if ((k == 4)&&(rule_init/8 == 1))
                    domain[i][j] = 1;
                if ((k == 3)&&((rule_init / 4) % 2 == 1))
                    domain[i][j] = 2;
                if ((k == 2)&&((rule_init / 2) % 2 == 1))
                    domain[i][j] = 3;
                if ((k == 1)&&(rule_init % 2 == 1))
                    domain[i][j] = 4;
            }
        }

        // 根据rule_ccend改变修改的序列
        if (rule_ccend == 1) {
            if (rule_gatc_avail % 2 == 1)
                domain[i][0] = 14;
            else if (rule_gatc_avail / 8 == 1)
                domain[i][0] = 11;
            else if ((rule_gatc_avail / 2) % 2 == 1)
                domain[i][0] = 13;
            else
                domain[i][0] = 12;

            if (rule_gatc_avail % 2  == 1)
                domain[i][domain_length[i]-1] = 14;
            else if (rule_gatc_avail / 8 == 1)
                domain[i][domain_length[i]-1] = 11;
            else if ((rule_gatc_avail / 4) % 2 == 1)
                domain[i][domain_length[i]-1] = 12;
            else
                domain[i][domain_length[i]-1] = 13;
        }
    }*/


//    // 重新随机序列 可能会用到
//    for (i = 0; i < num_domain; i++) {
//        for (j = 0; j < domain_length[i]; j++) {
//            if (domain[i][j] < 10) {
//                domain[i][j] = 0;
//                while (domain[i][j] == 0) {
//                    k = int_urn(1,4);
//                    if ((k == 4)&&(rule_init/8 == 1))
//                        domain[i][j] = 1;
//                    if ((k == 3)&&((rule_init / 4) % 2 == 1))
//                        domain[i][j] = 2;
//                    if ((k == 2)&&((rule_init / 2) % 2 == 1))
//                        domain[i][j] = 3;
//                    if ((k == 1)&&(rule_init % 2 == 1))
//                        domain[i][j] = 4;
//                }
//            }
//        }
//
//        if (rule_ccend == 1) {
//            if (domain_gatc_avail[i] % 2 == 1)
//                domain[i][0] = 14;
//            else if (domain_gatc_avail[i] / 8 == 1)
//                domain[i][0] = 11;
//            else if ((domain_gatc_avail[i] / 2) % 2 == 1)
//                domain[i][0] = 13;
//            else
//                domain[i][0] = 12;
//
//            if (domain_gatc_avail[i] % 2  == 1)
//                domain[i][domain_length[i]-1] = 14;
//            else if (domain_gatc_avail[i] / 8 == 1)
//                domain[i][domain_length[i]-1] = 11;
//            else if ((domain_gatc_avail[i] / 4) % 2 == 1)
//                domain[i][domain_length[i]-1] = 12;
//            else
//                domain[i][domain_length[i]-1] = 13;
//        }
//        gotoxy(6, 5+i);
//        for (j = 0; j < domain_length[i]; j++)
//            DisplayBase(domain[i][j]);
//    }

    // 用于后续计算
    // Set up domain score matrix
    // domain_score[] 分配num_domain个double代表 每个代表分数
    if ((domain_score = malloc(num_domain * sizeof(double))) == NULL) {
        fprintf(stderr, "Insufficient memory for declaring domain score matrix!\n");
        exit(-1);
    }

    // domain_intrinsic[] 分配num_domain个double代表 每个代表内部分数
    if ((domain_intrinsic = malloc(num_domain * sizeof(double))) == NULL) {
        fprintf(stderr, "Insufficient memory for declaring domain score matrix!\n");
        exit(-1);
    }

    // Set up crosstalk and interaction matrices
    // crosstalk[] 分配num_domain个double指针 记录分数; 修改了 只有第一个指针
    if ((crosstalk = malloc(1 * sizeof(double*))) == NULL) {
        fprintf(stderr, "Insufficient memory for declaring crosstalk matrices!\n");
        exit(-1);
    }

    // crosstalk[][] 分配num_domain个double值 记录分数; 修改了 只有第一个指针
    for (i = 0; i < 1; i++) {
        if ((crosstalk[i] = malloc(num_domain * sizeof(double))) == NULL) {
            fprintf(stderr, "Insufficient memory for declaring crosstalk matrices!\n");
            exit(-1);
        }
    }

    // interaction[] 分配num_domain个double指针; 修改了 只有第一个指针
    if ((interaction = malloc(1 * sizeof(double*))) == NULL) {
        fprintf(stderr, "Insufficient memory for declaring interaction matrices!\n");
        exit(-1);
    }

    // interaction[][] 分配num_domain个double值 记录分数; 修改了 只有分配一个
    for (i = 0; i < 1; i++) {
        if ((interaction[i] = malloc(num_domain * sizeof(double))) == NULL) {
            fprintf(stderr, "Insufficient memory for declaring interaction matrices!\n");
            exit(-1);
        }
    }

    //printf("compositions of each domain:\n");
    //for(i=0;i<num_domain;i++) printf("%s\n",char_domain[i]);

    // 评价分数  //这里修改了 只计算第一个探针的分数
    for (i = 0; i < 1; i++) {
        for (j = 0; j < num_domain; j++) {
            if (i == j) {
                // 不考虑自己 INTRA_SCORE是额外加分 所以也用不到self_crosstalk
                interaction[i][j] = 0;
                crosstalk[i][j] = 0;
            }
            else {
                interaction[i][j] = pairscore(domain[i], domain_length[i], domain[j], domain_length[j]);

                for (k = 0; k < domain_length[j]; k++)
                    temp_domain[k] = 15 - domain[j][(domain_length[j])-1-k];
                crosstalk[i][j] = pairscore(domain[i], domain_length[i], temp_domain, domain_length[j])/CROSSTALK_DIV;
            }
        }
    }

    for (i = 0; i < num_domain; i++)
        domain_intrinsic[i] = 0;

    // Search for 4g, if rule applied
    if (rule_4g == 1) {

        for (i = 0; i < num_domain; i++) {
            k = 0; // G-C counter
            for (j = 0; j < domain_length[i]; j++) {

                if ((domain[i][j] % 10 == 1)&&(k < 100))
                    k++;
                else if (domain[i][j] % 10 == 1)
                    k = 1;

                if ((domain[i][j] % 10 == 4)&&(k > 100))
                    k++;
                else if (domain[i][j] % 10 == 4)
                    k = 101;

                if ((k < 100)&&(k > 3))
                    domain_intrinsic[i] = domain_intrinsic[i] + GGGG_PENALTY;
                if (k > 103)
                    domain_intrinsic[i] = domain_intrinsic[i] + GGGG_PENALTY;
            }
        }
    }

    // Search for 6at, if rule applied
    if (rule_6at == 1) {
        for (i = 0; i < num_domain; i++) {
            k = 0; // AT counter
            for (j = 0; j < domain_length[i]; j++) {
                if ((domain[i][j] % 10 == 2)||(domain[i][j] % 10 == 3))
                    k++;
                else
                    k = 0;
                if (k > 5)
                    domain_intrinsic[i] = domain_intrinsic[i] + ATATAT_PENALTY;
            }

            k = 0; // GC counter
            for (j = 0; j < domain_length[i]; j++) {
                if ((domain[i][j] % 10 == 1)||(domain[i][j] % 10 == 4))
                    k++;
                else
                    k = 0;
                if (k > 5)
                    domain_intrinsic[i] = domain_intrinsic[i] + ATATAT_PENALTY;
            }
        }
    }

    // Domain score is max of interaction and crosstalk scores
    score = 0;
    // 修改了这里 使其只计算第一条序列的分数
    for (i = 0; i < 1; i++) {
        domain_score[i] = 0;
        for (j = 0; j < num_domain; j++) {
            if (interaction[i][j] + domain_importance[i] + domain_importance[j] > domain_score[i])
                domain_score[i] = interaction[i][j] + domain_importance[i] + domain_importance[j];
            if (crosstalk[i][j] + domain_importance[i] + domain_importance[j] + CROSSTALK_SCORE > domain_score[i])
                domain_score[i] = crosstalk[i][j] + domain_importance[i] + domain_importance[j] + CROSSTALK_SCORE;
        }
        domain_score[i] = domain_score[i] + domain_intrinsic[i] + (double) (i+1) * 0.000001;
        if (domain_score[i] > score) {
            score = domain_score[i];
            worst_domain = i;
        }
    }

    printf("%f",domain_score[0]); //只输出第一条
    return 0;
}
