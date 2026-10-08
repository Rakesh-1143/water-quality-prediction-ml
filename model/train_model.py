"""Run the paper-aligned experiment suite. Future-work features are excluded."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.model_selection import ParameterGrid, train_test_split
from xgboost import XGBClassifier, XGBRegressor
import shap
from config import (DATA_PATH, ARTIFACT_DIR, REPORT_DIR, FEATURES,
                    RANDOM_STATE, EPOCHS, XGB_PARAMS, ABSTRACT_ARTIFACT_DIR, ABSTRACT_REPORT_DIR)
from wqi.core import (load_data, split_data, preprocessor, shap_importance,
                      average, metrics, dataset_profile)
from wqi.neural import build_ann, fit_ann, probability


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x:x.item(), allow_nan=False))


def fit_experiment(X, y, seed, epochs, tune=False, initialization='repeat'):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    Xtr, Xv, Xt, ytr, yv, yt = split_data(X, y, seed)
    prep = preprocessor()
    tr, va, te = prep.fit_transform(Xtr), prep.transform(Xv), prep.transform(Xt)
    # Section III-C describes an initial XGBoost regressor for Tree SHAP.
    ranker = XGBRegressor(random_state=seed, n_jobs=2)
    ranker.fit(tr, ytr)
    explainer = shap.TreeExplainer(ranker)
    raw_shap = np.asarray(explainer.shap_values(tr))
    importance, weights = shap_importance(raw_shap)
    params = dict(XGB_PARAMS)
    ann_params = dict(learning_rate=.001, dropout=.2)
    tuning = {'enabled': tune, 'xgboost': [], 'ann': []}
    if tune:
        # Paper III-G: 20% of training data reserved for tuning. Outer
        # validation is reserved for ANN early stopping and fusion selection.
        ia, ib = train_test_split(np.arange(len(ytr)), test_size=.2,
            random_state=seed, stratify=ytr)
        inner = preprocessor()
        itr = inner.fit_transform(Xtr.iloc[ia]); iva = inner.transform(Xtr.iloc[ib])
        inner_ranker = XGBRegressor(random_state=seed, n_jobs=2).fit(itr, ytr.iloc[ia])
        _, inner_weights = shap_importance(shap.TreeExplainer(inner_ranker).shap_values(itr))
        for candidate in ParameterGrid({'max_depth':[3,5,6], 'n_estimators':[100,200],
                                       'learning_rate':[.05,.1]}):
            cfg = {**params, **candidate}
            m = XGBClassifier(**cfg, random_state=seed).fit(itr, ytr.iloc[ia])
            score = metrics(ytr.iloc[ib], m.predict_proba(iva)[:,1])['accuracy']
            tuning['xgboost'].append({'params':candidate, 'accuracy':score})
        params.update(max(tuning['xgboost'], key=lambda z:z['accuracy'])['params'])
        for candidate in ParameterGrid({'learning_rate':[.0005,.001], 'dropout':[.1,.2]}):
            m = build_ann(inner_weights, seed, initialization=initialization, **candidate)
            fit_ann(m, itr, ytr.iloc[ia], iva, ytr.iloc[ib], epochs, seed=seed)
            score = metrics(ytr.iloc[ib], probability(m, iva))['accuracy']
            tuning['ann'].append({'params':candidate, 'accuracy':score})
        ann_params.update(max(tuning['ann'], key=lambda z:z['accuracy'])['params'])
    xgb = XGBClassifier(**params, random_state=seed).fit(tr, ytr)
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=2).fit(tr,ytr)
    svr = SVR(kernel='rbf').fit(tr,ytr)
    ann = build_ann(weights, seed, initialization=initialization, **ann_params)
    history = fit_ann(ann, tr, ytr, va, yv, epochs, seed=seed)
    standard = build_ann(None, seed, **ann_params)
    standard_history = fit_ann(standard, tr, ytr, va, yv, epochs, seed=seed)
    pn, px = probability(ann, va), xgb.predict_proba(va)[:,1]
    ps = probability(standard, va)
    combiner = LogisticRegression(random_state=seed).fit(np.column_stack([pn,px]),yv)
    standard_combiner = LogisticRegression(random_state=seed).fit(np.column_stack([ps,px]),yv)
    # Both variants retained. Model/alpha selection uses validation only.
    candidates = {f'average_{a:.1f}':average(pn,px,a) for a in [.3,.4,.5,.6,.7]}
    candidates['stacking'] = combiner.predict_proba(np.column_stack([pn,px]))[:,1]
    validation = {n:metrics(yv,p) for n,p in candidates.items()}
    selected = max(validation, key=lambda n:validation[n]['accuracy'])
    alpha = float(selected.split('_')[1]) if selected.startswith('average_') else .5
    def predict_all(matrix):
        n, b = probability(ann,matrix), xgb.predict_proba(matrix)[:,1]
        s = probability(standard,matrix)
        return {'random_forest':rf.predict_proba(matrix)[:,1],
            'svr':np.clip(svr.predict(matrix),0,1), 'ann_xavier':s,
            'xgboost':b, 'ann_shap':n, 'hybrid_average_0.5':average(n,b,.5),
            'hybrid_stacking':combiner.predict_proba(np.column_stack([n,b]))[:,1],
            'hybrid_xavier_stacking':standard_combiner.predict_proba(np.column_stack([s,b]))[:,1],
            'hybrid_selected':average(n,b,alpha) if selected.startswith('average_') else
                combiner.predict_proba(np.column_stack([n,b]))[:,1]}
    predictions = predict_all(te)
    result = dict(seed=seed, initialization=initialization,
        split_sizes={'train':len(ytr),'validation':len(yv),'test':len(yt)},
        split_indices={'train':Xtr.index.tolist(),'validation':Xv.index.tolist(),'test':Xt.index.tolist()},
        selected_fusion=selected, alpha=alpha,
        test={n:metrics(yt,p) for n,p in predictions.items()},
        fusion_validation=validation, tuning=tuning,
        parameters={'xgboost':params,'ann':ann_params},
        history=history, standard_history=standard_history,
        feature_importance=dict(zip(FEATURES,importance.tolist())),
        initialization_weights=dict(zip(FEATURES,weights.tolist())))
    return dict(result=result, prep=prep, ann=ann, xgb=xgb, combiner=combiner,
        ranker=ranker, explainer=explainer, train_shap=raw_shap,
        tr=tr, te=te, Xt=Xt, yt=yt, predictions=predictions, predict_all=predict_all)


def plots(run, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    result = run['result']
    for name, values in run['predictions'].items():
        fpr,tpr,_ = roc_curve(run['yt'], values)
        plt.plot(fpr,tpr,label=f"{name}: {result['test'][name]['roc_auc']:.3f}")
    plt.plot([0,1],[0,1],'--',color='gray'); plt.xlabel('False positive rate')
    plt.ylabel('True positive rate'); plt.legend(fontsize=6); plt.tight_layout()
    plt.savefig(directory/'roc_curves.png'); plt.close()
    h=result['history']; plt.plot(h['loss'],label='Training'); plt.plot(h['val_loss'],label='Validation')
    plt.xlabel('Epoch'); plt.ylabel('Binary cross-entropy'); plt.legend(); plt.tight_layout()
    plt.savefig(directory/'loss_curves.png'); plt.close()
    scores=pd.DataFrame(result['test']).T[['accuracy','f1','rmse','mae']]
    scores.plot.bar(figsize=(12,5)); plt.tight_layout(); plt.savefig(directory/'metrics.png'); plt.close()
    for name in ['ann_shap','xgboost','hybrid_selected']:
        plt.scatter(run['yt'],run['predictions'][name],label=name,alpha=.2,s=12)
    plt.xlabel('Actual binary label'); plt.ylabel('Predicted probability'); plt.legend(); plt.tight_layout()
    plt.savefig(directory/'actual_vs_probability.png'); plt.close()
    shap.summary_plot(run['train_shap'],run['tr'],feature_names=FEATURES,show=False)
    plt.tight_layout(); plt.savefig(directory/'training_tree_shap.png'); plt.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=EPOCHS)
    parser.add_argument('--seed',type=int,default=RANDOM_STATE)
    parser.add_argument('--tune',action='store_true',help='Run explicit inner-training grids')
    parser.add_argument('--full',action='store_true',help='Add five-seed sensitivity and final-model SHAP')
    parser.add_argument('--initialization', choices=['repeat', 'feature_glorot'], default='repeat',
                        help='Paper vector expansion or abstract feature-weighted random initialization')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--reports',type=Path)
    args=parser.parse_args()
    abstract = args.initialization == 'feature_glorot'
    args.output = args.output or (ABSTRACT_ARTIFACT_DIR if abstract else ARTIFACT_DIR)
    args.reports = args.reports or (ABSTRACT_REPORT_DIR if abstract else REPORT_DIR)
    if args.epochs < 1: parser.error('--epochs must be positive')
    args.output.mkdir(parents=True,exist_ok=True); args.reports.mkdir(parents=True,exist_ok=True)
    X,y=load_data(DATA_PATH)
    dump(args.reports/'dataset_profile.json',dataset_profile(X,y))
    print('Training SHAP ANN, baselines, and both hybrid variants...',flush=True)
    run=fit_experiment(X,y,args.seed,args.epochs,args.tune,args.initialization)
    dump(args.reports/'evaluation.json',run['result'])
    pd.DataFrame(run['result']['test']).T.to_csv(args.reports/'metrics.csv')
    # Ablations isolate both initialization and hybridization.
    ablation_names=['ann_xavier','ann_shap','xgboost','hybrid_average_0.5',
                    'hybrid_stacking','hybrid_xavier_stacking']
    dump(args.reports/'ablations.json',{n:run['result']['test'][n] for n in ablation_names})
    sensitivity={}
    for feature in ['ph','Sulfate','Conductivity']:
        for delta in [-.1,.1]:
            perturbed=run['Xt'].copy(); perturbed[feature]*=1+delta
            preds=run['predict_all'](run['prep'].transform(perturbed))['hybrid_selected']
            scores=metrics(run['yt'],preds)
            scores['accuracy_change']=scores['accuracy']-run['result']['test']['hybrid_selected']['accuracy']
            sensitivity[f'{feature}:{delta:+.1f}']=scores
    dump(args.reports/'feature_sensitivity.json',sensitivity)
    # Save predictions with original row IDs for independent verification.
    pd.DataFrame({'row_id':run['Xt'].index,'actual':run['yt'].to_numpy(),
        **run['predictions']}).to_csv(args.reports/'test_predictions.csv',index=False,
                                     float_format='%.17g')
    joblib.dump(run['prep'],args.output/'preprocess.joblib')
    joblib.dump(run['combiner'],args.output/'combiner.joblib')
    run['ann'].save(args.output/'ann.keras'); run['xgb'].save_model(args.output/'xgboost.json')
    import importlib.metadata as package_metadata
    versions={name:package_metadata.version(name) for name in
        ['tensorflow-cpu','xgboost','shap','scikit-learn','numpy','pandas']}
    dump(args.output/'metadata.json',dict(schema=1,features=FEATURES,
        fusion='stacking' if run['result']['selected_fusion']=='stacking' else 'average',
        alpha=run['result']['alpha'],seed=args.seed,epochs_limit=args.epochs,
        initialization=args.initialization,
        dataset_sha256=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),versions=versions,
        test_metrics=run['result']['test']['hybrid_selected'],
        initialization_weights=run['result']['initialization_weights']))
    plots(run,args.reports)
    if args.full:
        seeds=list(range(args.seed,args.seed+5)); scores=[]
        for seed in seeds:
            print(f'Sensitivity seed {seed}',flush=True)
            repeat=run if seed==args.seed else fit_experiment(X,y,seed,args.epochs,args.tune,args.initialization)
            scores.append({'seed':seed,**repeat['result']['test']['hybrid_selected']})
        dump(args.reports/'split_sensitivity.json',{'runs':scores,
            'accuracy_std':float(np.std([s['accuracy'] for s in scores])),
            'f1_std':float(np.std([s['f1'] for s in scores]))})
        # Explain the actual hybrid probability, not only its XGBoost component.
        background=shap.kmeans(run['tr'],10)
        hybrid=lambda matrix:run['predict_all'](matrix)['hybrid_selected']
        final_explainer=shap.KernelExplainer(hybrid,background)
        examples=run['te'][:32]
        vals=np.asarray(final_explainer.shap_values(examples,nsamples=512))
        np.savez(args.reports/'hybrid_shap.npz',values=vals,features=examples)
        dump(args.reports/'hybrid_shap_importance.json',
             dict(zip(FEATURES,np.abs(vals).mean(axis=0).tolist())))
        import matplotlib.pyplot as plt
        shap.summary_plot(vals,examples,feature_names=FEATURES,show=False)
        plt.tight_layout(); plt.savefig(args.reports/'hybrid_shap.png'); plt.close()
        # Check alignment, without asserting domain expectations as facts.
        init=np.array(list(run['result']['initialization_weights'].values()))
        from scipy.stats import spearmanr
        rank_correlation=spearmanr(init,np.abs(vals).mean(axis=0)).statistic
        dump(args.reports/'importance_alignment.json',{'spearman':
            float(rank_correlation) if np.isfinite(rank_correlation) else None,
            'explained_test_rows':len(examples),'background_clusters':10})
    print(json.dumps(run['result']['test']['hybrid_selected'],indent=2),flush=True)
    print(f'Saved {args.initialization} artifacts to {args.output} and reports to {args.reports}.',flush=True)


if __name__=='__main__': main()
