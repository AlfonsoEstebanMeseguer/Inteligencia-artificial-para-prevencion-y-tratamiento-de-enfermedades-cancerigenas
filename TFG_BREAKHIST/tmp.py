from pathlib import Path  
lines=Path('cnn_transfer_2.py').read_text(encoding='utf-8').splitlines()  
for i,line in enumerate(lines[120:200],120): print('{}: {}'.format(i+1,line)) 
