import os

neutralDir = "./sourceData/neutral/"
happyDir = "./sourceData/happy/"
fearDir = "./sourceData/fear/"

trig = "./raw/ID10_Sessao1_N_triggers"
file = "./raw/ID10_Sessao1_N"

with open(trig,"r") as f:
    lines = f.readlines()
i=0
data = []
start = float(lines[3].replace("\n",""))
# end = float(lines[4].replace("\n",""))
end = start + 1000 * 1 * 60
with open(file,"r") as f:
    while True:
        if i <= start :
            i+=1
            f.readline()
            continue
        if i > end:
            break
        data += [f.readline()]
        i+=1
   
with open("data.csv", "w") as f:

    f.write("ecg,emgz,emg,eda\n")
    for line in data:
        f.write(line)




