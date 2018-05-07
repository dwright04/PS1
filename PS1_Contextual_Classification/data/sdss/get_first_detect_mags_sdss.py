detection_limts = {"u":22.0,"g":22.2,"r":22.2,"i":21.3,"z":20.5}
filter_keys ={"0":"u","1":"g","2":"r","3":"i","4":"z"}
path = "/Users/dew/development/PS1-Class/data/sdss/data/"

input = open(path+"files.txt","r")

sne = {}
for line in input.readlines():
    if "ReadMe" in line:
        continue
    phot = open(path+line.rstrip(),"r")
    for point in phot.readlines():
        point  = point.replace(" ", ",").replace(",,,",",").replace(",,",",")
        if "#" in point :
            continue
        if point[0] == ",":
            point = point[1:]
        filter = point.split(",")[2]
        mag = float(point.split(",")[3])
        if mag < detection_limts[filter_keys[filter]]:
            print line.split(".")[0], filter_keys[filter], mag
            sne[line.split(".")[0]] = {"mag":mag, "filter":filter}
            break