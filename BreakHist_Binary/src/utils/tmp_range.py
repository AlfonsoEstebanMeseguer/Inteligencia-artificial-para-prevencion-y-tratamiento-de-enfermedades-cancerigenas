lines = open('eda.py','r',encoding='utf-8',errors='ignore').read().splitlines()  
for i in range(1258, 1272):  
    print('{}: {}'.format(i, lines[i-1].encode('ascii','ignore').decode('ascii'))) 
