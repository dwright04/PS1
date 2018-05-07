import sys, os, urllib, urlparse, pyfits, pickle, optparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def DataDownload(remotedir, filename, localdir="./"):
    remoteaddr = 'http://%s%s' % (remotedir, filename)
    (scheme, server, path, params, query, frag) = urlparse.urlparse(remoteaddr)
    localname = os.path.split(path)[1]
    #print remoteaddr, localname
    try:
        # retrieve remoteaddr from server and store in localname on client
        urllib.urlretrieve(remoteaddr, localdir+localname+".txt")
    except IOError, e:
        print "ERROR: Failed to download. Error is: %s"% e.errno

def checkWebInfo(id, image_id, mjd, pathToMags, survey):
    if survey == "3pi":
        url = "http://star.pst.qub.ac.uk/sne/ps13pi/psdb/candidate/" + id + "/"
        servername = "star.pst.qub.ac.uk/sne/ps13pi/psdb/lightcurve/"
    elif survey == "md":
        url = "http://star.pst.qub.ac.uk/sne/ps1md/psdb/candidate/" + id + "/"
        servername = "star.pst.qub.ac.uk/sne/ps1md/psdb/lightcurve/"
    elif survey == "old_md":
        url = "http://star.pst.qub.ac.uk/ps1/psdb/candidate/" + id + "/"
        servername = "star.pst.qub.ac.uk/ps1/psdb/lightcurve/"    
    html = urllib.urlopen(url).read()
    index = html.index("Object List")
    objectList = html[index:index+50]
    index = html.index("Spectral Type:")
    type = html[index+22:index+30].strip("</h3><").strip("</h3")
    index = html.index("PS1 Name:")
    name = html[index+17:index+26].strip("<").strip("</")
    try:
        index = html.index("<td>"+image_id)
        mag = html[index-80:index-70].strip(" <td>").strip("</")
        assert (float(mag) < 25.0 and float(mag) > 0)
    except Exception, e:
        print e
        filename = id+".txt"
        try:
            open(pathToMags+filename, "r")
        except:
            DataDownload(servername, id, pathToMags)
        for line in open(pathToMags+filename, "r").readlines():
            if "#" in line or line.rstrip() == "":
                continue
            print line.strip()
            if (float(line.rstrip().split(" ")[0][:9]) - mjd) < 1e-7:
                if line.rstrip().split(" ")[1] == "None":
                    continue
                mag = float(float(line.rstrip().split(" ")[1]))
            else:
                mag = 0

    if type == "d":
        index = html.index("Contextual Classification:")
        type = "("+html[index+33:index+50].strip("</h3></div>\n").strip(">")+")"
    if "good" in objectList:
        return "good", type, name, mag
    elif "garbage" in objectList:
        return "garbage", type, name, mag
    elif "confirmed" in objectList:
        return "confirmed", type, name, mag
    elif "possible" in objectList:
        return "possible", type, name, mag
    elif "attic" in objectList:
        return "attic", type, name, mag
        
def web_info(to_porcess, survey, output_file):
    
    path2Images = "/Users/dew/myscripts/machine_learning/data/%s/detectionlist/" % survey
    path2Real = "2/"
    path2Confirmed = "1/"
    path2Bogus = "0/"
    pathToMags = "/Users/dew/development/PS1-Real-Bogus/data/%s/mags/" % survey

    counter = 1
    mags = []
    output = open(output_file, "a")
    for file in to_porcess:
        id = file.split("_")[0]
        try:
            hdulist = pyfits.open(path2Images+path2Real + file)
            imagefile = path2Images+path2Real + file
        except:
            try:
                hdulist = pyfits.open(path2Images+path2Confirmed + file)
                imagefile = path2Images+path2Confirmed + file
            except:
                hdulist = pyfits.open(path2Images+path2Bogus + file)
                imagefile = path2Images+path2Bogus + file
        #hdulist = pyfits.open(pathToImages+image)
        header = hdulist[1].header
        #print id
        #print header.keys()
        #sys.exit()
        image_id = file.split("_diff.fits")[0]
        #print image_id
        mjd = hdulist[1].header["MJD-OBS"]

        print checkWebInfo(id, image_id, mjd, pathToMags, survey)
        objectList, type, name, mag = checkWebInfo(id, image_id, mjd, pathToMags, survey)
        try:
            #print mag
            mag = float(mag)
        except Exception, e:
            print e
            print id
            print image_id
            print mag
        if "PS1" not in name or name == "h3></di":
            name = id
            #print counter, name + " & " + str(mjd) + " & " + type + " & " + \
            #      header["HIERARCH FPA.FILTERID"].split(".")[0] + " & " + \
            #      str(mag) + " & " + objectList + " & " + \
            #      str(pred[subset1[i]]) + " \\\\"
        counter+=1
        mags.append(mag)
        print name, mag
        output.write(name + "," + str(mjd) + "," + type + "," + \
                     header["HIERARCH FPA.FILTERID"].split(".")[0] + "," + \
                     str(mag) + "," + objectList + "," + imagefile + "\n")
    output.close()
    
    plt.hist(mags, bins=50)
    plt.show()

def main():
    parser = optparse.OptionParser("[!] usage: python get_web_info.py\n"+\
                                   "\t -F <data file>\n"+\
                                   "\t -s <survey [3pi, md, old_md] >\n")
                                   
    parser.add_option("-F", dest="dataFile", type="string", \
                      help="specify data file to get web info for")
    parser.add_option("-s", dest="survey", type="string", \
                      help="specify which survey this is for [3pi, md, old_md]")
                      
    (options, args) = parser.parse_args()
    dataFile = options.dataFile
    survey = options.survey
    
    if dataFile == None or survey == None:
    	print parser.usage
        exit(0)
        
    output_file = "web_info_%s.csv" % dataFile.split("/")[-1].split(".")[0]
    
    to_process = []

    try:
        data = sio.loadmat(dataFile)
        # get training image file names
        try:
            to_process = to_process + list(data["train_files"])
        except KeyError:
            try:
                to_process = to_process + list(data["images"])
            except KeyError:
                sys.exit("[!] no files found for training set.")
        # get validation set file names
        try:
            to_process = to_process + list(data["valid_files"])
        except KeyError:
            print "[*] Warning: no files found for validation set."
        # get test set file names
        try:
            to_process = to_process + list(data["test_files"])
        except KeyError:
            print "[*] Warning: no files found for test set."
    except ValueError:
        try:
           for line in open(dataFile,"r").readlines():
                to_process.append(line.rstrip())
        except IOError:
            sys.exit("[!] Could not open %s" % dataFile)
    
    try:
        assert survey in set(["3pi", "md", "old_md"])
    except AssertionError:
        sys.exit("[!] survey must be one of [3pi, md, old_md]")
    
    print "[+] Getting web info for %d files ..." % (len(to_process))
        
    str_to_process = [str(x).rstrip() for x in to_process]

    if dataFile.split(".")[-1] == "mat":
        try:
            check_log = open(output_file,"r")
            for line in check_log.readlines():
                str_to_process.remove(line.rstrip().split(",")[-1].split("/")[-1])
            check_log.close()
        except IOError:
            pass

    print "[*] Already processed %d files" % (len(to_process) - len(str_to_process))
    to_process = None
    web_info(str_to_process, survey, output_file)
        
if __name__ == "__main__":
    main()
