"""Losslessly compress bulky event traces before private-network retrieval."""
import gzip,pathlib,shutil,sys
folder=pathlib.Path(sys.argv[1])
for pattern in ['probe-snapshot-*.json','profile-raw-*.json']:
 for path in folder.glob(pattern):
  target=path.with_suffix(path.suffix+'.gz'); temp=target.with_suffix(target.suffix+'.tmp')
  with path.open('rb') as src,gzip.open(temp,'wb',compresslevel=1) as dst:shutil.copyfileobj(src,dst)
  temp.replace(target);path.unlink()
