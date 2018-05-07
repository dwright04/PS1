import optparse, sys, subprocess, json, time, os
import urllib, urlparse

from bs4 import BeautifulSoup

"""
    RUN ON PSDB
"""
catalogues = ["tcs_cat_v_sdss_dr9_stars", \
              "tcs_cat_sdss_dr9_spect_galaxies_qsos", \
              "tcs_cat_v_sdss_dr9_galaxies_notspec"]

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
    except:
        pass

def get_host_info(id, ra, dec):
    sdss_host_id = "NULL"
    sdss_host_offset = 99.0
    sdss_host_cat = None
    for catalogue in catalogues:
        cmd = "ConeSearch dew \"\" panstarrs1 psdb2 quick %s %s %s 5" % \
              (catalogue, ra, dec)
        try:
            result = subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError, e:
            """ subprocess exit status != 0 """
            return "NULL", "NULL", "NULL", "NULL"
        result = result.split("\n\t")
        try:
            result = result[1]
        except IndexError:
            pass
        if "ID" in result:
            tmp_id = result.split(" ")[1].strip(",")
            tmp_offset = float(result.split(" ")[4][:-2])
            if tmp_offset <= sdss_host_offset:
                sdss_host_id = tmp_id
                sdss_host_offset = tmp_offset
                sdss_host_cat = catalogue

    if sdss_host_cat == catalogues[0]:
        is_gal = False
        is_star = True
    elif sdss_host_cat in catalogues[1:]:
        is_gal = True
        is_star = False
    else:
        is_gal = False
        is_star = False
    return sdss_host_id, sdss_host_offset, is_gal, is_star

def get_class(url):
    SN = ["I","Ia", "Ia pec", "Ib", "Ibc", "Ic", "II", "IIb", "IIn", "II-L", "II-P", "II pec"]
    STAR = ["stellar", "CV"]
    AGN = ["AGN / QSO"]
    try:
        html = urllib.urlopen(url).read()
    except IOError, e:
        print "ERROR: Failed to download. Error is: %s"% e.errno
        return "NULL"
    soup = BeautifulSoup(html)
    for line in soup.find_all('h3'):
        if "Spectral Type" in str(line):
            print str(line).split("</span>")[-1].strip("</h3>")
            classification = str(line).split("</span>")[-1].strip("</h3>")
            if classification in SN:
                return "SN"
            elif classification in STAR:
                return "STAR"
            elif classification in AGN:
                return "AGN"
            else:
                return "Other"
    return "NULL"

def get_first_detect_mag(url, id):
    filename = id+".txt"
    DataDownload(url, id, "sdss_hosts/3pi/tmp/")
    for line in open("sdss_hosts/3pi/tmp/"+filename, "r").readlines():
        if "#" in line or line.rstrip() == "":
            continue
        print line.strip()
        if line.rstrip().split(" ")[1] == "None":
                continue
        mag = float(line.rstrip().split(" ")[1])
        filter = line.rstrip().split(" ")[5]
    try:
        return mag, filter
    except:
        return "NULL", "NULL"

def get_host_mags(id):
    query = "'select r, err_r, g, err_g, u, err_u, i, err_i, z, err_z from PhotoObj where objID = %s'" % id
    cmd = "python sqlcl.py -q %s" % query
    try:
        result = subprocess.check_output(cmd, shell=True).split("\n")[-2].split(",")
        #print result
        if result[0] == "No objects have been found":
            return "NULL", "NULL", "NULL", "NULL","NULL", "NULL", "NULL", "NULL","NULL", "NULL"
        return result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8], result[9], result[10]
    except subprocess.CalledProcessError, e:
        """ subprocess exit status != 0 """
        return "NULL", "NULL", "NULL", "NULL","NULL", "NULL", "NULL", "NULL","NULL", "NULL"
    except IndexError:
        return "NULL", "NULL", "NULL", "NULL","NULL", "NULL", "NULL", "NULL","NULL", "NULL"

def get_host_photoz(id):
    query = "'select z, zErr from Photoz where objID = %s'" % id
    cmd = "python sqlcl.py -q %s" % query
    try:
        result = subprocess.check_output(cmd, shell=True).split("\n")[-2].split(",")
        #print result
        if result[0] == "No objects have been found":
            return 0, 0
        return result[1], result[2]
    except subprocess.CalledProcessError, e:
        """ subprocess exit status != 0 """
        return 0, 0
    except IndexError:
        return 0, 0

def get_host_qso_colours(id):
    query = "'select legacy_target1 from SpecPhotoAll where objID = %s'" % id
    cmd = "python sqlcl.py -q %s" % query
    try:
        result = subprocess.check_output(cmd, shell=True).split("\n")[-2].split(",")
        #print result
        if result[0] == "No objects have been found":
            return False
        #print int(result[1]) & (0 | 2 | 4 | 8 | 16) > 0
        return int(result[1]) & (0 | 2 | 4 | 8 | 16) > 0
    except subprocess.CalledProcessError, e:
        """ subprocess exit status != 0 """
        return False
    except IndexError:
        return False

def main():
    
    
    parser = optparse.OptionParser("[!] usage: python scrape_contextual_features.py\n"+\
                                   "\t -i <input file [comma-separated]>\n"+\
                                   "\t -s <survey [3pi, md, old_md, fgss, old_fgss]>")

    parser.add_option("-i", dest="inputFile", type="string", \
                      help="specify input file. File must contain comma-separated id, RA and Dec.")
    parser.add_option("-s", dest="survey", type="string", \
                      help="specify survey. Choose 1 of [3pi, md, old_md, fgss, old_fgss]")
        

    (options, args) = parser.parse_args()

    inputFile = options.inputFile
    survey = options.survey

    if inputFile == None or survey == None:
        print parser.usage
        exit(0)
    
    if survey == "3pi":
        base_url = "http://star.pst.qub.ac.uk/sne/ps13pi/psdb/candidate/"
    elif survey == "md":
        base_url = "http://star.pst.qub.ac.uk/sne/ps1md/psdb/candidate/"
    elif survey == "old_md":
        base_url = "http://star.pst.qub.ac.uk/ps1/psdb/candidate/"
    elif survey == "fgss":
        base_url = "http://star.pst.qub.ac.uk/sne/ps1fgss/psdb/candidate/"
    elif survey == "old_fgss":
        base_url = "http://star.pst.qub.ac.uk/ps1fgss/psdb/candidate/"

    try:
        input = open(inputFile, "r")
    except IOError:
        sys.exit("[!] Could not open file : %s" % inputFile)

    features ={"rows":[]}
    for line in input.readlines():
        data = line.rstrip().split(",")
        print data
        object = {"transient_object_id":data[0], "ra":data[1], \
                  "dec":data[2], "survey":survey}
        object["class"] = get_class(base_url+data[0])
        print object["class"]
        try:
            sdss_host_id, sdss_host_offset, is_gal, is_star = \
            get_host_info(data[0], data[1], data[2])
        except TypeError:
            """NoneType is not iterable """
            continue
        print base_url[6:].replace("candidate", "lightcurve")
        object["first_detect_mag"], object["first_detect_filter"] = \
        get_first_detect_mag(base_url[7:].replace("candidate", "lightcurve"), data[0])
        object["sdss_host"] = sdss_host_id
        object["sdss_host_offset"] = sdss_host_offset
        object["is_gal"] = is_gal
        object["is_star"] = is_star
        r, err_r, g, err_g, u, err_u, i, err_i, z, err_z = get_host_mags(sdss_host_id)
        object["u"] = u
        object["err_u"] = err_u
        object["g"] = g
        object["err_g"] = err_g
        object["r"] = r
        object["err_r"] = err_r
        object["i"] = i
        object["err_i"] = err_i
        object["z"] = z
        object["err_z"] = err_z
        photoz, zErr = get_host_photoz(sdss_host_id)
        object["photoz"] = photoz
        object["err_photoz"] = zErr
        has_qso_colours = get_host_qso_colours(sdss_host_id)
        object["has_qso_colours"] = has_qso_colours
        print object
        features["rows"].append(object)
    #print features
    with open("output.json", "w") as output:
         output.write(json.dumps(features, indent=4))
    output.close()



if __name__ == "__main__":
    main()
