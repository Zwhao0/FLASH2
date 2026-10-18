"""Private-network helper; credentials are read from ignored 服务器.md, never logged."""
import argparse, concurrent.futures, pathlib, re, sys
import paramiko
ROOT = pathlib.Path(__file__).resolve().parents[1]
def connect(host):
    content = (ROOT / '服务器.md').read_text()
    password = re.search(r'密码\s+(\S+)', content).group(1)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username='test', password=password, timeout=12)
    return client
def run(host, cmd):
    c=connect(host)
    _, out, err=c.exec_command(cmd)
    for line in out: print(f'[{host}] {line}', end='', flush=True)
    error=err.read().decode()
    if error: print(f'[{host} stderr] {error}', file=sys.stderr)
    code=out.channel.recv_exit_status()
    c.close()
    return code
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--host',default='all'); p.add_argument('command'); a=p.parse_args()
    hosts=['10.10.4.1','10.10.4.2'] if a.host=='all' else [a.host]
    with concurrent.futures.ThreadPoolExecutor() as pool:
        codes=list(pool.map(lambda h:run(h,a.command),hosts))
    sys.exit(max(codes))
