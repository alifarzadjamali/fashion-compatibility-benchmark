import json, os, sys
L, D, out = sys.argv[1:4]
def auc(y, s):
    pos=[x for x,l in zip(s,y) if l==1]; neg=[x for x,l in zip(s,y) if l==0]
    r=sorted((v,i) for i,v in enumerate(s)); ranks=[0]*len(s); i=0
    while i<len(r):
        j=i
        while j+1<len(r) and r[j+1][0]==r[i][0]: j+=1
        for k in range(i,j+1): ranks[r[k][1]]=(i+j)/2+1
        i=j+1
    rp=sum(ranks[i] for i,l in enumerate(y) if l==1)
    return (rp-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def fitb_acc(f):
    sc=f['candidate_scores']; ans=f.get('answers', f.get('labels'))
    return sum(max(range(len(c)),key=lambda j:c[j])==a for c,a in zip(sc,ans))/len(sc)
rows=[]
for n in sorted(os.listdir(L)):
    a=json.load(open(os.path.join(L,n))); b=json.load(open(os.path.join(D,n)))
    ka=[k for k in a['cp'] if k!='labels'][0]
    ds=max(abs(x-y) for x,y in zip(a['cp'][ka],b['cp'][ka]))
    if a.get('fitb') and b.get('fitb'):
      fs=max(abs(x-y) for ca,cb in zip(a['fitb']['candidate_scores'],b['fitb']['candidate_scores']) for x,y in zip(ca,cb))
    else: fs=0.0
    aa,ab=auc(a['cp']['labels'],a['cp'][ka]),auc(b['cp']['labels'],b['cp'][ka])
    sel_same=a['selection'].get('chosen')==b['selection'].get('chosen')
    fitb_same=1.0 if not a.get('fitb') else sum(x==y for x,y in zip(a['fitb']['chosen_index'],b['fitb']['chosen_index']))/len(a['fitb']['chosen_index']); rows.append(dict(file=n,fitb_choice_agreement=fitb_same,cp_max_abs=ds,fitb_max_abs=fs,auc_laptop=aa,auc_dgx=ab,auc_diff=abs(aa-ab),selection_same=sel_same))
s=dict(n_files=len(rows),max_cp_score_diff=max(r['cp_max_abs'] for r in rows),max_fitb_score_diff=max(r['fitb_max_abs'] for r in rows),
       max_auc_diff=max(r['auc_diff'] for r in rows),min_fitb_choice_agreement=min(r['fitb_choice_agreement'] for r in rows),selection_mismatches=[r['file'] for r in rows if not r['selection_same']],
       worst=sorted(rows,key=lambda r:-r['auc_diff'])[:5])
json.dump(dict(summary=s,rows=rows),open(out,'w'),indent=1)
print(json.dumps({k:v for k,v in s.items() if k!='worst'},indent=1)); print([ (w['file'],round(w['auc_diff'],5)) for w in s['worst']])
