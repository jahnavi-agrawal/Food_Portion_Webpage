"""Report prediction MAE/RMSE from a prediction JSON and `instance_id,weight_g` CSV."""
import argparse, csv, json, math
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('prediction'); p.add_argument('ground_truth'); a=p.parse_args()
 pred=json.loads(Path(a.prediction).read_text())['foods']; truth=list(csv.DictReader(open(a.ground_truth, newline='')))
 errors=[float(x['weight_g'])-float(y['weight_g']) for x,y in zip(pred,truth)]
 print(json.dumps({'n':len(errors),'mae_g':sum(map(abs,errors))/len(errors),'rmse_g':math.sqrt(sum(x*x for x in errors)/len(errors))},indent=2))
if __name__=='__main__': main()
