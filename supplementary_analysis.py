"""제출 전 보완 분석: Train 진단·전처리 ablation·고정 모델 간 탐색적 AP 신뢰구간."""
from pathlib import Path
import ast
import json
import time

ROOT = Path(__file__).resolve().parent


def plot_train_risk(risk, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    for ax, grouping in zip(axes, ['pastdue','income_observed']):
        sub = risk[risk.grouping==grouping]
        if grouping == 'pastdue':
            sub = sub.set_index('group').reindex(['0','1-2','3-5','Other normal','Special 96/98']).reset_index()
        ax.bar(sub['group'],sub['positive_rate'],color='#193449')
        ax.set_title(f'Train positive rate: {grouping}')
        ax.tick_params(axis='x',rotation=25)
        ax.set_ylabel('Positive rate')
        ax.set_ylim(0,float(sub['positive_rate'].max())*1.25)
        for bar, n in zip(ax.patches,sub['n']):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.005,f'n={n:,}',ha='center',fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'train_group_risk.png',dpi=200,bbox_inches='tight')
    plt.close(fig)


def add_zero_income_flag(X):
    result = X.copy()
    result['Zero_Income_flag'] = (result['Income'] == 0).astype(int)
    return result


def load_context():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    SEED = 33
    data = pd.read_csv(ROOT/'givemesomecredit.csv').rename(columns={
        'SeriousDlqin2yrs': 'Dlq2yrs', 'RevolvingUtilizationOfUnsecuredLines': 'RevUtil',
        'age': 'Age', 'NumberOfTime30-59DaysPastDueNotWorse': 'PastDue_30-59',
        'MonthlyIncome': 'Income', 'NumberOfDependents': 'Dependents',
        'CombinedDefaulted': 'CombDefault', 'CombinedCreditLoans': 'CombCredit',
    })
    split = np.load(ROOT/'split_indices.npz')
    X = data.drop(columns='Dlq2yrs')
    y = data['Dlq2yrs']
    nb = json.loads((ROOT/'ass.ipynb').read_text(encoding='utf-8'))
    scope = dict(locals())
    scope.update(globals())
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(c['source'])).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'models' for t in node.targets):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<baseline model settings>', 'exec'), scope)
    return X.loc[split['train_idx']], y.loc[split['train_idx']], X.loc[split['test_idx']], y.loc[split['test_idx']], scope['models']


def run_supplementary():
    import numpy as np
    import pandas as pd
    import matplotlib
    if __name__ == '__main__':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.base import clone
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import average_precision_score
    from imblearn.pipeline import Pipeline
    from gmsc_preprocessing import GMSCPreprocessor

    out = ROOT/'submission/supplementary'
    out.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_test, y_test, models = load_context()
    frozen_hash = __import__('hashlib').sha256((ROOT/'submission/final_model.joblib').read_bytes()).hexdigest()
    train = X_train.copy()
    train['target'] = y_train
    groupings = {
        'pastdue': np.select([train['PastDue_30-59'].isin([96,98]), train['PastDue_30-59']==0,
                              train['PastDue_30-59'].between(1,2), train['PastDue_30-59'].between(3,5)],
                             ['Special 96/98', '0', '1-2', '3-5'], default='Other normal'),
        'debt_ratio': pd.cut(train['DebtRatio'], [-np.inf,1,10,100,1000,np.inf],
                            labels=['<=1','1-10','10-100','100-1000','>1000']),
        'rev_util': pd.cut(train['RevUtil'], [-np.inf,1,10,100,np.inf], labels=['<=1','1-10','10-100','>100']),
        'income_observed': np.select([train['Income']==0, train['Income']==6600],
                                     ['Income=0', 'Income=6600 (observed)'], default='Other income'),
    }
    risk_rows = []
    for grouping, group in groupings.items():
        tmp = train.assign(group=group)
        for label, sub in tmp.groupby('group', observed=True, sort=False):
            risk_rows.append({'grouping':grouping, 'group':str(label), 'n':len(sub),
                              'positives':int(sub['target'].sum()), 'positive_rate':float(sub['target'].mean()),
                              'debt_ratio_median':float(sub['DebtRatio'].median()),
                              'debt_ratio_p95':float(sub['DebtRatio'].quantile(.95))})
    risk = pd.DataFrame(risk_rows)
    risk.to_csv(out/'train_group_risk.csv', index=False)
    plot_train_risk(risk, out)

    # Raw도 공통 스케일링을 적용한다. Clean 조건과의 차이는 특수값 치환+flag의 묶음 효과다.
    options = [('Raw8',None,False),('Clean_NoLog',False,False),
               ('Clean_Log',True,False),('Clean_Log_ZeroIncome',True,True)]
    cv_splits = list(StratifiedKFold(5,shuffle=True,random_state=33).split(X_train,y_train))
    folds = []
    for name in ['LR','XGB']:
        for label, use_log, zero_flag in options:
            print(f'Ablation: {name} / {label}',flush=True)
            pipe = Pipeline([
                ('income_flag',FunctionTransformer(add_zero_income_flag) if zero_flag else 'passthrough'),
                ('clean',GMSCPreprocessor(use_log=use_log) if use_log is not None else 'passthrough'),
                ('scale',StandardScaler()), ('model',clone(models[name])),
            ])
            score = cross_validate(pipe,X_train,y_train,cv=cv_splits,
                                   scoring={'AP':'average_precision','AUROC':'roc_auc'},
                                   n_jobs=3,pre_dispatch=3,error_score='raise')
            for i in range(5):
                folds.append({'model':name,'preprocessing':label,'fold':i+1,
                              'n_features':8 if use_log is None else 9+int(zero_flag),
                              'AP':score['test_AP'][i],'AUROC':score['test_AUROC'][i]})
    ablation_folds = pd.DataFrame(folds)
    ablation = ablation_folds.groupby(['model','preprocessing'],sort=False).agg(
        AP_mean=('AP','mean'),AP_std=('AP','std'),AUROC_mean=('AUROC','mean'),AUROC_std=('AUROC','std')).reset_index()
    ablation.to_csv(out/'preprocessing_ablation.csv',index=False)
    ablation_folds.to_csv(out/'preprocessing_ablation_folds.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,metric in zip(axes,['AP_mean','AUROC_mean']):
        ablation.pivot(index='model',columns='preprocessing',values=metric).reindex(columns=[r[0] for r in options]).plot(kind='bar',ax=ax,rot=0)
        ax.set_title(metric)
        ax.legend_.remove()
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,fontsize=9)
    fig.tight_layout(rect=[0,0,1,.87])
    fig.savefig(out/'preprocessing_ablation.png',dpi=200,bbox_inches='tight')
    plt.close(fig)

    # 초기 CV 순위의 후보와 같은 설정을 전체 Train에 학습한다. 튜닝 XGB와 혼동하지 않는다.
    probs = {}
    for label, name, use_ir in [('XGB_None','XGB',False),('LGB_None','LGB',False),('XGB_IR','XGB',True)]:
        model = clone(models[name])
        model.set_params(scale_pos_weight=float((y_train==0).sum()/(y_train==1).sum()) if use_ir else 1.0)
        pipe=Pipeline([('clean',GMSCPreprocessor(use_log=True)),('scale',StandardScaler()),('model',model)])
        pipe.fit(X_train,y_train)
        probs[label]=pipe.predict_proba(X_test)[:,list(pipe.classes_).index(1)]
    pd.DataFrame({'row_index':X_test.index,'y_true':y_test.to_numpy(),**probs}).to_csv(out/'bootstrap_predictions.csv',index=False)
    truth=y_test.to_numpy()
    positive=np.flatnonzero(truth==1)
    negative=np.flatnonzero(truth==0)
    rng=np.random.default_rng(33)
    pairs=[('XGB_None','LGB_None'),('XGB_None','XGB_IR')]
    draws={f'{a} - {b}':[] for a,b in pairs}
    for i in range(1000):
        # 같은 재추출 행을 모든 모델에 적용하고 관측 클래스 비율을 유지한다.
        idx=np.concatenate([rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)])
        scores={k:average_precision_score(truth[idx],v[idx]) for k,v in probs.items()}
        for a,b in pairs:
            draws[f'{a} - {b}'].append(float(scores[a]-scores[b]))
        if (i+1)%250==0:
            print(f'Paired bootstrap {i+1}/1000',flush=True)
    cis=[]
    for a,b in pairs:
        key=f'{a} - {b}'
        low,high=np.quantile(draws[key],[.025,.975])
        cis.append({'comparison':key,'AP_A':average_precision_score(truth,probs[a]),
                    'AP_B':average_precision_score(truth,probs[b]),
                    'delta_AP':average_precision_score(truth,probs[a])-average_precision_score(truth,probs[b]),
                    'CI_low':float(low),'CI_high':float(high),'contains_zero':bool(low<=0<=high),
                    'n_bootstrap':1000})
    ci=pd.DataFrame(cis)
    ci.to_csv(out/'paired_bootstrap_ci.csv',index=False)
    pd.DataFrame(draws).to_csv(out/'paired_bootstrap_draws.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,3))
    for i,row in ci.iterrows():
        ax.plot([row.CI_low,row.CI_high],[i,i],color='#193449',linewidth=3)
        ax.scatter(row.delta_AP,i,color='#199C9C',zorder=3)
    ax.axvline(0,color='gray',linestyle='--')
    ax.set_yticks(range(len(ci)),ci['comparison'])
    ax.set_xlabel('Delta AP (A - B), paired stratified percentile 95% CI')
    ax.set_title('Exploratory comparison of fixed fitted baseline models')
    fig.tight_layout()
    fig.savefig(out/'paired_bootstrap_ci.png',dpi=200,bbox_inches='tight')
    plt.close(fig)
    assert __import__('hashlib').sha256((ROOT/'submission/final_model.joblib').read_bytes()).hexdigest()==frozen_hash
    report={'seed':33,'n_bootstrap':1000,'stratified':True,'model_unchanged_sha256':frozen_hash,
            'risk_groups':risk_rows,'ablation':ablation.to_dict('records'),'bootstrap':cis,
            'limitations':[
                '추가 진단과 ablation은 최초 Test 확인 후 수행한 탐색적 분석이며 최종 모델 재선정에 사용하지 않았다.',
                'DebtRatio 극단값의 모든 원인이 무소득 또는 절대액 기록이라고 단정하지 않는다.',
                'Income=6600은 관측값 그룹이며 원래 결측이었다는 표식이 아니다.',
                'bootstrap은 고정된 기준 모델의 Test 표본 불확실성이다. CV 평균 차이의 CI가 아니며 재학습·모델선택 변동성을 포함하지 않는다.',
                '관측 클래스 비율에 조건부인 층화 bootstrap의 percentile 구간이며 사후 두 비교에 대한 동시 신뢰구간/다중검정 보정이 아니다.',
            ]}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(ablation.to_string(index=False),flush=True)
    print(ci.to_string(index=False),flush=True)
    return report


if __name__=='__main__':
    import sys
    if '--plots-only' in sys.argv:
        import matplotlib
        matplotlib.use('Agg')
        import pandas as pd
        out=ROOT/'submission/supplementary'
        plot_train_risk(pd.read_csv(out/'train_group_risk.csv'),out)
    else:
        run_supplementary()
