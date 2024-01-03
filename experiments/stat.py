import pandas as pd
import numpy as np
from scipy import stats


#等分散性の確認（F検定）、有意水準5%、帰無仮説を「対象の2群間の分散に差はないこと」とする
def f_test(A, B):
    A_var = np.var(A, ddof=1)
    B_var = np.var(B, ddof=1)
    A_df = len(A)-1
    B_df = len(B)-1
    f = A_var / B_var
    one_sided_pval1 = stats.f.cdf(f, A_df, B_df)
    one_sided_pval2 = stats.f.sf(f, A_df, B_df)
    two_sided_pval = min(one_sided_pval1, one_sided_pval2) * 2
    return f, two_sided_pval

df = pd.read_csv('statictics.csv', index_col = 0, header=0, engine='python')
cases = df.index

result = pd.DataFrame([],columns=['A', 'B', 'Different', 'normality A', 'normality B', 'shapiro value A', 'shapiro value B', 'f value', 'f p', 'test method', 't value', 't p'])

for i, A in enumerate(cases):
    for j, B in enumerate(cases):
        if i>=j:
            continue
        case = A + ' vs ' +B
        a = df.loc[A] #Series of mean values of caseA
        a = a.iloc[0:10].values
        result.loc[case, 'A'] = A #Setting name
        b = df.loc[B]
        b = b.iloc[0:10].values
        result.loc[case, 'B'] = B #Setting name
        #shapiro-wilk test 帰無仮説は「母集団が正規分布である」
        _, wpa = stats.shapiro(a)
        result.loc[case, 'shapiro value A'] = wpa
        result.loc[case, 'normality A'] = True if wpa > 0.05 else False
        _, wpb = stats.shapiro(b)
        result.loc[case, 'shapiro value B'] = wpb
        result.loc[case, 'normality B'] = True if wpb > 0.05 else False
        #F-test
        f, fp = f_test(a,b)
        result.loc[case, 'f value'] = f
        result.loc[case, 'f p'] = fp
        if fp > 0.05: #帰無仮説は棄却されず、2群間は等分散であること（少なくとも不等分散ではないこと）が示唆された
            #Student t_test
            test = stats.ttest_ind(a, b)
            s='student'
        else:
            #Welch t_test
            #2群間の平均値の差を比較する検定
            #2群間の平均値が独立であり（データに対応がない）、2群間に等分散性が仮定できない場合
            test = stats.ttest_ind(a, b, equal_var=False)
            s='Welch'
        result.loc[case, 'test method'] = s
        result.loc[case, 't value'] = test[0]
        result.loc[case, 't p'] = test[1]
        if test[1]<0.05:#帰無仮説は棄却され、2群間に差があると言える
            #print (A, 'and', B, 'is possibly different by', s, 'test', ', ', test)
            result.loc[case, 'Different'] = True
        else:
            #print (A, 'and', B, 'is possibly same by', s, 'test', ', ', test)
            result.loc[case, 'Different'] = False
            
result.to_csv('t_test_result.csv')
