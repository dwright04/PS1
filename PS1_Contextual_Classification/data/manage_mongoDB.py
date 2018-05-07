import json, pprint
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pymongo import MongoClient#, Connection
#from astropy import coordinates as coord
from astropy.coordinates import SkyCoord

def main():

    #"pessto/pessto_features.txt" excluded as large array of filter sets.
    
    spec_dataFiles = ["md/MD_ps1ss_features.txt", \
                      "md/MD_features.txt",\
                      "3pi/3pi_features.txt",\
                      "fgss/FGSS_features_dr9.txt",\
                      "fgss/OLDFGSS_features_dr9.txt",\
                      "sdss/SDSSSN_features.txt",\
                      "psst/PSST_features.txt"]

    phot_dataFiles = ["md/test_data/MDphotAGN_features.txt",\
                      "3pi/test_data/3pi_features_test.txt",\
                      "3pi/test_data/3pi_confirmed_since21Mar2015.txt"]

    #dataFiles = ["psst/PSST_features.txt"]
    #c = Connection()
    c = MongoClient()
    c.drop_database("detection_context_features_spec")
    #c.drop_database("detection_context_features_phot")

    client = MongoClient('localhost:27017')
    db = client.detection_context_features_spec
    #db = client.detection_context_features_phot

    keys = ["u", "err_u", "g", "err_g", "r", "err_r", "i", "err_i", "z", "err_z", "ra"]
    #keys = ["ra"]
    for dataFile in spec_dataFiles:
    #for dataFile in phot_dataFiles:
    #for dataFile in dataFiles:
    #    print dataFile
        open(dataFile, "r")
        data = json.load(open(dataFile, "r"))["rows"]
    #    print data
        error_count = 0
        for element in data:
            for key in keys:
                if key == "ra":
                    try:
                        element["ra"] = float(str(element["ra"]))
                    except ValueError:
                        try:
                            ra = element["ra"].replace(":", "h",1).replace(":", "m",1) + "s"
                            dec = element["dec"].replace(":", "d",1).replace(":", "m",1) + "s"
                            element["ra"] = SkyCoord(ra+" "+dec).ra.deg
                            element["dec"] = SkyCoord(ra+" "+dec).dec.deg
                        except AttributeError:
                            element["ra"] = None
                            element["dec"] = None
                else:
                    try:
                        element[key] = float(str(element[key]))
                    except ValueError as e:
                        print("error ")
                        print(e)
                        print(str(element[key]))
                        error_count += 1
                        pass
                    except KeyError:
                        error_count += 1
                        pass
            print(element)
            db.detection_context_features_spec.insert(element)
        print(error_count)
            #db.detection_context_features_phot.insert(element)
    #    #for line in open(dataFile, "r"):
    #    #    print line

    """

    query = {"$and": [{"host_r":{"$gte":21}},\
                      {"host_r":{"$ne":"NULL"}},\
                      {"class":{"$ne":"NULL"}},\
                      {"class":{"$ne":"Other"}},\
                      {"ra":{"$ne":None}},\
                      {"dec":{"$ne":None}},\
                      {"sdss_host_offset":{"$ne":None}}]}

    #query = {"survey":{"$eq":"3pi"}}
    projection = {"_id" : 0, "transient_object_id": 1, "class" : 1, \
                  "sdss_host_offset" : 1, "has_qso_colours" : 1, "is_star" : 1, \
                  "is_gal" : 1, "sdss_host" : 1, "host_r" : 1, "survey" : 1, \
                  "photoz" : 1, "err_photoz" : 1, "ra" : 1, "dec" : 1}
    #results = db.detection_context_features.find(query, projection)
    results = db.detection_context_features_spec.find(query, projection)
    print results
    #num_results = db.detection_context_features.find(query, projection).count()
    num_results = db.detection_context_features_spec.find(query, projection).count()
    print num_results
    X = np.ones((num_results, 8)) # 8 features
    y = np.ones((num_results,))
    ids = []
    surveys = []
    for i,result in enumerate(results):
        print i,result["survey"], result["ra"]
        if result["sdss_host_offset"] != "NULL" :
            X[i,0] *= result["sdss_host_offset"]
            X[i,1] *= result["host_r"]
            X[i,2] *= result["is_gal"]
            X[i,3] *= result["has_qso_colours"]
            X[i,6] *= result["ra"]
            X[i,7] *= result["dec"]
            try:
                X[i,4] *= result["photoz"]
                X[i,5] *= result["err_photoz"]
            except ValueError: 
                X[i,4] *= 0
                X[i,5] *= 0
            ids.append(result["transient_object_id"])
            surveys.append(result["survey"])
            if result["class"] == "SN":
                y[i] *= 1
            elif result["class"] == "AGN":
                y[i] *= 0
            elif result["class"] == "STAR":
                y[i] *= 2
            else:
               print "[!] " + str(result)
        else:
            print "[!] " + str(result)
    print len(np.where(y==1)[0])
    print len(np.where(y==0)[0])
    print len(np.where(y==2)[0])

    #sio.savemat("context_data_set_maglt21_8features_3pi_test.mat", {"X":X, "y":y, "ids":ids, "surveys":surveys})
    """
    #bins = [x for x in np.arange(0, 6.2, 0.2)]
    """
    agn_counts, bins, patches = plt.hist(agn_offsets, bins=bins, label="AGNS : %d" % len(agn_offsets), \
                                                                 normed="True", \
                                                                 color="#FF0066", \
                                                                 edgecolor="none")
    """
    """    
    sn_counts, bins, patches = plt.hist(sn_offsets, bins=bins, label="SNE : %d" % len(sn_offsets), \
                                                               normed="True", \
                                                               color="#66FF33", \
                                                               edgecolor="none") 
    """
    """
    star_counts, bins, patches = plt.hist(star_offsets, bins=bins, label="STARS : %d" % len(star_offsets),\
                                                                   normed="True", \
                                                                   color="#3366FF", \
                                                                   edgecolor="none")
    """
    #print "[+] %d AGNs." % len(agn_offsets)
    #print "[+] %d SNe."  % len(sn_offsets)
    #print "[+] %d STARS." % len(star_offsets)
    #sum = float(len(agn_offsets)+len(sn_offsets))
    #try:
    #    overlap = list(np.where(np.array(agn_counts) <= np.array(sn_counts))[0])
    #    for i in range(len(overlap)):
    #        to_plot = [bins[overlap[i]], bins[overlap[i]+1]]
    #        plt.hist(agn_counts, bins=to_plot, color="#FF0066", edgecolor="none")
    #except IndexError:
    #    pass
    #plt.xlabel("SDSS host offset [arcsecs]")
    #plt.ylabel("normalised counts")
    #plt.legend()
    #plt.show()
 
   
if __name__ == "__main__":
    main()
