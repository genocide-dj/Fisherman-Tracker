content = open('src/tracker.py').read() 
fixed = content.replace('tid % len(colors)', 'hash(tid) % len(colors)') 
open('src/tracker.py', 'w').write(fixed) 
print('Fixed') 
