"""Real-data entry points for the four extensions; no automatic fake labels."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier
import shap
from config import FEATURES, XGB_PARAMS
from wqi.core import shap_importance, preprocessor
from wqi.validation import model_inputs
from .core import (features, labeled, partitions, seed_all, fit, scores, binary_prob,
                   fingerprint, reject_overlap, save, validate_time, training_preprocessor)


def multiclass(frame, output, label='WQI_class', epochs=100, seed=42):
    seed_all(seed)
    X, y, classes = labeled(frame, label, True)
    tr, va, te = partitions(y, seed)
    prep = training_preprocessor(X.iloc[tr])
    a,b,c = [model_inputs(prep.transform(X.iloc[z])) for z in [tr,va,te]]
    params = {**XGB_PARAMS, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss',
              'num_class': len(classes), 'random_state': seed}
    tree = XGBClassifier(**params).fit(a, y[tr])
    _, weights = shap_importance(shap.TreeExplainer(tree).shap_values(a))
    model = tf.keras.Sequential([tf.keras.layers.Input((9,)),
        tf.keras.layers.Dense(16, activation='relu', name='feature_layer'),
        tf.keras.layers.Dropout(.2), tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(len(classes), activation='softmax')])
    base = tf.keras.initializers.GlorotUniform(seed=seed)((9,16)).numpy()
    model.get_layer('feature_layer').set_weights([base*np.sqrt(9*weights)[:,None], np.zeros(16)])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss='sparse_categorical_crossentropy')
    history = fit(model, a, y[tr], b, y[va], epochs, seed)
    nv = model(b.astype('float32'), training=False).numpy()
    xv = tree.predict_proba(b)
    validations = {str(alpha): scores(y[va], alpha*nv+(1-alpha)*xv, classes)
                   for alpha in [.3,.4,.5,.6,.7]}
    alpha = float(max(validations, key=lambda z: validations[z]['accuracy']))
    nn = model(c.astype('float32'), training=False).numpy()
    xp = tree.predict_proba(c)
    p = alpha*nn+(1-alpha)*xp
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=2).fit(a,y[tr])
    report = {'test': {'ann': scores(y[te],nn,classes), 'xgboost': scores(y[te],xp,classes),
        'random_forest': scores(y[te],rf.predict_proba(c),classes), 'hybrid': scores(y[te],p,classes)},
        'validation': validations, 'history': history,
        'split_indices': dict(zip(['train','validation','test'],[z.tolist() for z in [tr,va,te]]))}
    predictions = pd.DataFrame({f'probability_{name}':p[:,i] for i,name in enumerate(classes)})
    predictions.insert(0,'row_id',te); predictions.insert(1,'actual',np.array(classes)[y[te]])
    save(output, model, prep, {'kind':'multiclass','classes':classes,'alpha':alpha,'label':label,
         'seed':seed,'dataset_sha256':fingerprint(frame),'initialization_weights':weights.tolist()},report,predictions)
    tree.save_model(Path(output)/'xgboost.json')
    return report


def timeseries(frame, output, label='WQI', timestamp='timestamp', lookback=12,
               horizon=1, epochs=100, seed=42):
    seed_all(seed)
    X = features(frame)
    stamps, interval = validate_time(frame, timestamp)
    if lookback < 2 or horizon < 1:
        raise ValueError('lookback >= 2 and horizon >= 1 are required')
    if label not in frame:
        raise ValueError('Measured WQI values are required; Potability is not WQI')
    y = pd.to_numeric(frame[label], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError('WQI targets must be finite observed values')
    n = len(frame); stop = int(n*.7); val_stop = int(n*.85)
    # Windows remain wholly inside each partition (conservative gap protocol).
    bounds = [(0,stop),(stop,val_stop),(val_stop,n)]
    if any(end-start < lookback+horizon+3 for start,end in bounds):
        raise ValueError('Each chronological partition needs enough rows for at least four windows')
    if X.iloc[:stop].isna().all().any():
        raise ValueError('Training partition has an unobserved feature')
    prep = training_preprocessor(X.iloc[:stop]); scaled = model_inputs(prep.transform(X))
    mean = float(y[:stop].mean()); std = float(y[:stop].std())
    if std <= 0:
        raise ValueError('Training WQI targets must vary')
    normalized = (y-mean)/std
    groups=[]
    for start,end in bounds:
        ids=np.arange(start+lookback+horizon-1,end)
        windows=np.array([scaled[t-horizon-lookback+1:t-horizon+1] for t in ids],dtype='float32')
        groups.append((windows,normalized[ids].astype('float32'),ids))
    model=tf.keras.Sequential([tf.keras.layers.Input((lookback,9)),
        tf.keras.layers.LSTM(32),tf.keras.layers.Dense(16,activation='relu'),tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='mse')
    history=fit(model,*groups[0][:2],*groups[1][:2],epochs,seed)
    xt,_,ids=groups[2]
    p=model(xt,training=False).numpy().reshape(-1)*std+mean
    def regression(pred):
        return {'mae':float(mean_absolute_error(y[ids],pred)),
                'rmse':float(np.sqrt(mean_squared_error(y[ids],pred)))}
    # Persistence uses only the latest known target before the forecast horizon.
    persistence=y[ids-horizon]
    report={'test':{'lstm':regression(p),'persistence':regression(persistence)},'history':history,
        'partition_bounds':bounds,'target_indices':[g[2].tolist() for g in groups],
        'protocol':'Disjoint chronological partitions; windows never cross a boundary.'}
    save(output,model,prep,{'kind':'timeseries','lookback':lookback,'horizon':horizon,
        'timestamp':timestamp,'label':label,'interval_ns':interval,'seed':seed,
        'target_mean':mean,'target_std':std,'dataset_sha256':fingerprint(frame)},report,
        pd.DataFrame({'row_id':ids,'timestamp':stamps.iloc[ids].astype(str).to_numpy(),
                      'actual':y[ids],'prediction':p,'persistence':persistence}))
    return report


def transfer(frame, source, output, epochs=100, seed=42):
    seed_all(seed)
    source=Path(source)
    meta=json.loads((source/'metadata.json').read_text())
    if meta.get('schema') != 1 or meta.get('features') != FEATURES:
        raise ValueError('Use a nine-feature binary source ANN from model/abstract or model/paper')
    X,y,classes=labeled(frame)
    from config import DATA_PATH
    import hashlib
    if meta.get('dataset_sha256') == hashlib.sha256(DATA_PATH.read_bytes()).hexdigest():
        reject_overlap(pd.read_csv(DATA_PATH),frame)
    tr,va,te=partitions(y,seed)
    # Keep the source preprocessing coordinate system unchanged during transfer.
    prep=joblib.load(source/'preprocess.joblib')
    a,b,c=[model_inputs(prep.transform(X.iloc[z])) for z in [tr,va,te]]
    model=tf.keras.models.load_model(source/'ann.keras',compile=False)
    if model.input_shape != (None,9) or model.output_shape != (None,1):
        raise ValueError('Source model must be the compatible binary ANN')
    zero_shot=binary_prob(model,c)
    for layer in model.layers[:-1]: layer.trainable=False
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='binary_crossentropy')
    warmup=fit(model,a,y[tr].astype('float32'),b,y[va].astype('float32'),min(epochs,10),seed)
    warm_weights=model.get_weights()
    warm_score=float(
        tf.keras.losses.binary_crossentropy(y[va,None].astype('float32'),model(b.astype('float32'),training=False)).numpy().mean())
    for layer in model.layers: layer.trainable=True
    model.compile(optimizer=tf.keras.optimizers.Adam(.0001),loss='binary_crossentropy')
    history=fit(model,a,y[tr].astype('float32'),b,y[va].astype('float32'),epochs,seed)
    fine_score=float(tf.keras.losses.binary_crossentropy(y[va,None].astype('float32'),model(b.astype('float32'),training=False)).numpy().mean())
    if fine_score > warm_score: model.set_weights(warm_weights)
    p=binary_prob(model,c)
    report={'test':{'source_ann_zero_shot':scores(y[te],zero_shot,classes),
                    'transferred_ann':scores(y[te],p,classes)},'warmup_history':warmup,'fine_tuning_history':history,
            'selected_stage':'warmup' if fine_score>warm_score else 'fine_tuning',
            'split_indices':dict(zip(['train','validation','test'],[z.tolist() for z in [tr,va,te]]))}
    save(output,model,prep,{'kind':'transfer','classes':classes,'seed':seed,
        'source_model_sha256':hashlib.sha256((source/'ann.keras').read_bytes()).hexdigest(),
        'target_dataset_sha256':fingerprint(frame)},report,
        pd.DataFrame({'row_id':te,'actual':y[te],'potable_probability':p[:,1]}))
    return report


@tf.keras.utils.register_keras_serializable(package='WaterQuality')
class GradientReversal(tf.keras.layers.Layer):
    def __init__(self, strength=1., **kwargs):
        super().__init__(**kwargs); self.strength=float(strength)

    def call(self, inputs):
        strength=self.strength
        @tf.custom_gradient
        def reverse(x):
            return tf.identity(x), lambda gradient: -strength*gradient
        return reverse(inputs)

    def get_config(self):
        return {**super().get_config(),'strength':self.strength}


def dann(source, target_adapt, output, target_test=None, epochs=100, seed=42, strength=1.):
    seed_all(seed)
    if epochs < 1 or not np.isfinite(strength) or strength <= 0:
        raise ValueError('Positive epochs and finite gradient strength are required')
    X,y,classes=labeled(source); T=features(target_adapt)
    reject_overlap(source,target_adapt)
    if target_test is not None:
        reject_overlap(target_adapt,target_test); reject_overlap(source,target_test)
        E,ey,_=labeled(target_test, evaluation=True)
    tr,va,te=partitions(y,seed)
    prep=training_preprocessor(X.iloc[tr])
    a,b,c=[model_inputs(prep.transform(X.iloc[z])) for z in [tr,va,te]]
    t=model_inputs(prep.transform(T))
    inputs=tf.keras.Input((9,)); shared=tf.keras.layers.Dense(32,activation='relu')(inputs)
    shared=tf.keras.layers.Dense(16,activation='relu')(shared)
    label=tf.keras.layers.Dense(1,activation='sigmoid',name='label')(shared)
    domain=tf.keras.layers.Dense(1,activation='sigmoid',name='domain')(GradientReversal(strength)(shared))
    full=tf.keras.Model(inputs,[label,domain]); predictor=tf.keras.Model(inputs,label)
    optimizer=tf.keras.optimizers.Adam(.001); bce=tf.keras.losses.BinaryCrossentropy()
    rng=np.random.default_rng(seed); best=np.inf; wait=0; best_weights=full.get_weights(); history=[]
    for epoch in range(epochs):
        losses=[]
        for batch in np.array_split(rng.permutation(len(a)),max(1,int(np.ceil(len(a)/32)))):
            target_ids=rng.integers(len(t),size=len(batch))
            matrix=np.concatenate([a[batch],t[target_ids]])
            domains=np.concatenate([np.zeros((len(batch),1)),np.ones((len(batch),1))]).astype('float32')
            with tf.GradientTape() as tape:
                lp,dp=full(matrix,training=True)
                # Only labeled SOURCE rows contribute to the class loss.
                class_loss=bce(y[tr[batch],None].astype('float32'),lp[:len(batch)])
                loss=class_loss+bce(domains,dp)
            grads=tape.gradient(loss,full.trainable_variables)
            optimizer.apply_gradients(zip(grads,full.trainable_variables)); losses.append(float(loss))
        val=float(bce(y[va,None].astype('float32'),predictor(b,training=False)))
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'source_validation_loss':val})
        if val<best: best=val; best_weights=full.get_weights(); wait=0
        else: wait+=1
        if wait>=10: break
    full.set_weights(best_weights)
    p=binary_prob(predictor,c)
    report={'source_test':scores(y[te],p,classes),'history':history,
        'target_labels_used_for_training':False,
        'source_split_indices':dict(zip(['train','validation','test'],[z.tolist() for z in [tr,va,te]]))}
    if target_test is not None:
        ep=binary_prob(predictor,prep.transform(E)); report['target_test']=scores(ey,ep,classes)
    save(output,predictor,prep,{'kind':'dann','classes':classes,'seed':seed,'gradient_strength':strength,
        'source_dataset_sha256':fingerprint(source),'adaptation_dataset_sha256':fingerprint(target_adapt),
        'target_test_dataset_sha256':fingerprint(target_test) if target_test is not None else None},report,
        pd.DataFrame({'row_id':te,'actual':y[te],'potable_probability':p[:,1]}))
    if target_test is not None:
        pd.DataFrame({'row_id':range(len(E)),'actual':ey,'potable_probability':ep[:,1]}).to_csv(
            Path(output)/'target_test_predictions.csv',index=False,float_format='%.17g')
    return report
