// Docker builder only: preserve every required ELF library and its package evidence.
import fs from 'node:fs';import path from 'node:path';import {execFileSync} from 'node:child_process';
const root='/runtime-root';fs.mkdirSync(root,{recursive:true});
function copy(source){const dest=root+source;fs.mkdirSync(path.dirname(dest),{recursive:true});fs.copyFileSync(source,dest);}
copy('/usr/local/bin/node');
const linked=execFileSync('/usr/bin/ldd',['/usr/local/bin/node'],{encoding:'utf8'});if(linked.includes('not found'))throw Error('Missing Node ELF dependency');
const libraries=[...new Set([...linked.matchAll(/(?:=>\s+)?(\/[^\s]+)/g)].map(x=>x[1]))];for(const name of libraries)copy(name);
const owners=new Map();for(const entry of fs.readdirSync('/var/lib/dpkg/info').filter(x=>x.endsWith('.list'))){const pkg=entry.slice(0,-5).split(':')[0];for(const name of fs.readFileSync('/var/lib/dpkg/info/'+entry,'utf8').split('\n')){owners.set(name,pkg);if(name.startsWith('/usr/lib/'))owners.set(name.slice(4),pkg);if(name.startsWith('/lib/'))owners.set('/usr'+name,pkg);}}
const packages=new Set();for(const name of libraries){const pkg=owners.get(name)||owners.get(fs.realpathSync(name));if(!pkg)throw Error('Unattributed runtime dependency: '+name);packages.add(pkg);}
fs.cpSync('/etc/ssl/certs',root+'/etc/ssl/certs',{recursive:true,dereference:true});packages.add('ca-certificates');for(const name of ['/etc/os-release','/etc/debian_version','/etc/nsswitch.conf'])copy(name);
const paragraphs=fs.readFileSync('/var/lib/dpkg/status','utf8').split('\n\n').filter(p=>packages.has(p.match(/^Package: (.+)$/m)?.[1]));fs.mkdirSync(root+'/var/lib/dpkg',{recursive:true});fs.writeFileSync(root+'/var/lib/dpkg/status',paragraphs.join('\n\n')+'\n');for(const pkg of packages){const p='/usr/share/doc/'+pkg+'/copyright';if(fs.existsSync(p))copy(p);}
fs.writeFileSync(root+'/etc/passwd','captrail:x:10001:10001:CapTrail:/app:/sbin/nologin\n');fs.writeFileSync(root+'/etc/group','captrail:x:10001:\n');fs.mkdirSync(root+'/tmp',{mode:0o1777});
fs.writeFileSync(root+'/runtime-manifest.json',JSON.stringify({node:process.version,openssl:process.versions.openssl,versions:process.versions,libraries,packages:[...packages],policy:'Full ELF closure with retained Debian package and copyright evidence. Official Node embeds its own OpenSSL; unrelated Debian libssl is not copied. No vulnerability suppression.'},null,2));
