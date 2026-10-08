"""Train or predict with separately labeled experimental future-work models."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
from wqi.validation import read_csv
from future_work.training import multiclass, timeseries, transfer, dann
from future_work.core import FuturePredictor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['multiclass','timeseries','transfer','dann']:
        p=sub.add_parser(name)
        p.add_argument('--output',type=Path,required=True,help='A new empty directory')
        p.add_argument('--epochs',type=int,default=100)
        p.add_argument('--seed',type=int,default=42)
        if name!='dann': p.add_argument('--data',type=Path,required=True)
        if name=='multiclass': p.add_argument('--label',default='WQI_class')
        if name=='timeseries':
            p.add_argument('--label',default='WQI'); p.add_argument('--timestamp',default='timestamp')
            p.add_argument('--lookback',type=int,default=12); p.add_argument('--horizon',type=int,default=1)
        if name=='transfer': p.add_argument('--source-model',type=Path,default=Path('model/abstract'))
        if name=='dann':
            p.add_argument('--source',type=Path,required=True)
            p.add_argument('--target-adapt',type=Path,required=True)
            p.add_argument('--target-test',type=Path)
            p.add_argument('--strength',type=float,default=1.)
    p=sub.add_parser('predict')
    p.add_argument('--model',type=Path,required=True); p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        if args.command=='predict':
            if args.output.resolve()==args.data.resolve(): raise ValueError('Do not overwrite input measurements')
            prediction=FuturePredictor(args.model).predict(read_csv(args.data))
            args.output.parent.mkdir(parents=True,exist_ok=True)
            prediction.to_csv(args.output,index=False,float_format='%.17g')
            print(f'Saved {len(prediction)} prediction rows'); return
        if args.epochs<1: raise ValueError('Epochs must be positive')
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError('Choose a new empty output directory')
        common={'output':args.output,'epochs':args.epochs,'seed':args.seed}
        if args.command=='dann':
            report=dann(read_csv(args.source),read_csv(args.target_adapt),
                target_test=read_csv(args.target_test) if args.target_test else None,
                strength=args.strength,**common)
        else:
            frame=read_csv(args.data)
            if args.command=='multiclass': report=multiclass(frame,label=args.label,**common)
            elif args.command=='timeseries':
                report=timeseries(frame,label=args.label,timestamp=args.timestamp,
                                  lookback=args.lookback,horizon=args.horizon,**common)
            else: report=transfer(frame,args.source_model,**common)
        print(json.dumps(report.get('test',{'source_test':report.get('source_test'),
                                         'target_test':report.get('target_test')}),indent=2))
        print('Experimental run saved. Its scores do not establish paper reproduction.')
    except (ValueError,OSError,KeyError) as exc:
        parser.exit(1,f'Future-work command failed: {exc}\n')


if __name__=='__main__': main()
