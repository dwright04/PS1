import numpy as np
import scipy.io as sio

from pymongo import MongoClient
from sklearn import preprocessing

def build_data_set(results, num_results):
  
  X = np.ones((num_results, 9))
  y = np.ones((num_results,))
  ids = []
  surveys = []
  i=0
  for result in results:
  
    X[i,0] *= float(result["u"]) - float(result["g"])
    X[i,1] *= float(result["g"]) - float(result["r"])
    X[i,2] *= float(result["r"]) - float(result["i"])
    X[i,3] *= float(result["i"]) - float(result["z"])
    X[i,4] *= float(result["photoz"])
    X[i,5] *= float(result["is_gal"])
    X[i,6] *= float(result["has_qso_colours"])
    X[i,7] *= float(result["sdss_host_offset"])
    try:
      X[i,8] *= float(result["first_detect_mag"]) - \
            float(result[result["first_detect_filter"]])
    except KeyError:
      print('e')
      X[i,8] *= float(result["first_detect_mag"]) - \
        float(result["r"])
    
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
    i+=1
  return X, y, np.array(ids), surveys

def get_data():
  projection = {"_id" : 0, "transient_object_id": 1, "class" : 1,\
                "sdss_host_offset" : 1, "has_qso_colours" : 1,\
                "is_gal" : 1, "sdss_host" : 1, "survey" : 1,\
                "photoz" : 1, "err_photoz" : 1, "ra" : 1, "dec" : 1,\
                "first_detect_mag" : 1, "first_detect_filter" : 1,\
                "u" : 1, "g" : 1, "r" : 1, "i" : 1, "z" : 1}


  client = MongoClient('localhost:27017')
  db = client.detection_context_features_spec

  first_detect_mag = {}
  query = {"$and": [{"u":{"$ne":"NULL"}},\
            {"g":{"$ne":"NULL"}},\
            {"r":{"$ne":"NULL"}},\
            {"i":{"$ne":"NULL"}},\
            {"z":{"$ne":"NULL"}},\
            {"photoz":{"$ne":"NULL"}},\
            {"first_detect_mag":{"$ne":"NULL"}},\
            {"first_detect_filter":{"$ne":"NULL"}},\
            {"first_detect_filter":{"$ne":"y"}},\
            {"first_detect_filter":{"$ne":"w"}},\
            {"class":{"$ne":"NULL"}},\
            {"class":{"$ne":"Other"}},\
            {"class":{"$ne":"other"}},\
            {"ra":{"$ne":None}},\
            {"dec":{"$ne":None}},\
            {"photoz":{"$ne":None}},\
            {"sdss_host_offset":{"$ne":"NULL"}},\
            {"sdss_host_offset":{"$ne":None}}]}

  num_results = db.detection_context_features_spec.find(query, projection).count()
  print(num_results)
  results = db.detection_context_features_spec.find(query, projection)
  X, y, ids, surveys = build_data_set(results, num_results)
  print(np.array(ids))
  data = sio.loadmat("/Users/dwright/old_mac_home/development/PS1-Class/analysis/exploratory_analysis/contextual_classification_dataset_20150706_shuffled.mat")
  x = data["X"]
  print(data["y"])

  t = []
  for i in range(len(x)):
    for j in range(len(X)):
      if np.all(X[j]==x[i]):
        print(ids[j])
        t.append(ids[j])
  print(len(t))
  print(len(set(t)))
  #print(np.array(ids[np.where(X==x[0])]))

  print(data.keys())
def main():
  get_data()
if __name__ == '__main__':
  main()
