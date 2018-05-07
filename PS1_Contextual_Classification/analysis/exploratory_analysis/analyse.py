import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymongo import MongoClient, Connection
from NeuralNet import NeuralNet
import pickle
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

def main():

    projection = {"_id" : 0, "transient_object_id": 1, "class" : 1, \
                  "sdss_host_offset" : 1, "has_qso_colours" : 1, \
                  "is_gal" : 1, "sdss_host" : 1, "survey" : 1, \
                  "photoz" : 1, "err_photoz" : 1, "ra" : 1, "dec" : 1, \
                  "first_detect_mag" : 1, "first_detect_filter" : 1, \
                  "u" : 1, "g" : 1, "r" : 1, "i" : 1, "z" : 1}


    client = MongoClient('localhost:27017')
    db = client.detection_context_features_spec

    first_detect_mag = {}
    query = {"$and": [{"survey":{"$eq":"md"}},\
                      {"u":{"$ne":"NULL"}},\
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
    print num_results
    results = db.detection_context_features_spec.find(query, projection)
    #print results
    first_detect_mag["md"]=[]
    for result in results:
        #first_detect_mag["md"].append(float(result["first_detect_mag"]))
        first_detect_mag["md"].append(float(result["sdss_host_offset"]))
    num_results = db.detection_context_features_spec.find(query, projection).count()
    results = db.detection_context_features_spec.find(query, projection)
    X_md, y_md, ids_md, surveys = build_data_set(results, num_results)
    #print X_md
    print len(y_md)

    query = {"$and": [{"survey":{"$eq":"old_md"}},\
                  {"u":{"$ne":"NULL"}},\
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
    results = db.detection_context_features_spec.find(query, projection)
    first_detect_mag["oldmd"]=[]
    for result in results:
        #first_detect_mag["md"].append(float(result["first_detect_mag"]))
        first_detect_mag["oldmd"].append(float(result["sdss_host_offset"]))
    num_results = db.detection_context_features_spec.find(query, projection).count()
    results = db.detection_context_features_spec.find(query, projection)
    X_oldmd, y_oldmd, ids_oldmd, surveys = build_data_set(results, num_results)
    print len(y_oldmd)


    query = {"$and": [{"survey":{"$eq":"fgss"}},\
                  {"u":{"$ne":"NULL"}},\
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
    results = db.detection_context_features_spec.find(query, projection)
    first_detect_mag["fgss"]=[]
    for result in results:
        #first_detect_mag["md"].append(float(result["first_detect_mag"]))
        first_detect_mag["fgss"].append(float(result["sdss_host_offset"]))
    num_results = db.detection_context_features_spec.find(query, projection).count()
    results = db.detection_context_features_spec.find(query, projection)
    X_fgss, y_fgss, ids_fgss, surveys = build_data_set(results, num_results)
    print len(y_fgss)


    query = {"$and": [{"survey":{"$eq":"old_fgss"}},\
                  {"u":{"$ne":"NULL"}},\
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
    results = db.detection_context_features_spec.find(query, projection)
    #oldfgss_first_detect_mag = float(result["first_detect_mag"])
    first_detect_mag["oldfgss"]=[]
    for result in results:
        #first_detect_mag["md"].append(float(result["first_detect_mag"]))
        first_detect_mag["oldfgss"].append(float(result["sdss_host_offset"]))
    num_results = db.detection_context_features_spec.find(query, projection).count()
    results = db.detection_context_features_spec.find(query, projection)
    X_oldfgss, y_oldfgss, ids_oldfgss, surveys = build_data_set(results, num_results)
    print len(y_oldfgss)

    query = {"$and": [{"survey":{"$eq":"3pi"}},\
                  {"u":{"$ne":"NULL"}},\
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
    results = db.detection_context_features_spec.find(query, projection)
    first_detect_mag["3pi"]=[]
    for result in results:
        #first_detect_mag["md"].append(float(result["first_detect_mag"]))
        first_detect_mag["3pi"].append(float(result["sdss_host_offset"]))
    num_results = db.detection_context_features_spec.find(query, projection) .count()
    results = db.detection_context_features_spec.find(query, projection)
    X_3pi, y_3pi, ids_3pi, surveys = build_data_set(results, num_results)
    print len(y_3pi)

    query = {"$and": [{"survey":{"$eq":"sdss"}},\
                      {"u":{"$ne":"NULL"}},\
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
    results = db.detection_context_features_spec.find(query, projection)
    first_detect_mag["sdss"]=[]
    for result in results:
        #first_detect_mag["md"].append(float(result["first_detect_mag"]))
        first_detect_mag["sdss"].append(float(result["sdss_host_offset"]))
    num_results = db.detection_context_features_spec.find(query, projection).count()
    results = db.detection_context_features_spec.find(query, projection)
    X_sdss, y_sdss, ids_sdss, surveys = build_data_set(results, num_results)
    print len(y_sdss)

    query = {"$and": [{"survey":{"$eq":"psst"}},\
                  {"u":{"$ne":"NULL"}},\
                  {"g":{"$ne":"NULL"}},\
                  {"r":{"$ne":"NULL"}},\
                  {"i":{"$ne":"NULL"}},\
                  {"z":{"$ne":"NULL"}},\
                  {"photoz":{"$ne":"NULL"}},\
                  {"first_detect_mag":{"$ne":"NULL"}},\
                  {"first_detect_filter":{"$ne":"NULL"}},\
                  {"first_detect_filter":{"$ne":"y"}},\
                  {"class":{"$ne":"NULL"}},\
                  {"class":{"$ne":"Other"}},\
                  {"class":{"$ne":"other"}},\
                  {"ra":{"$ne":None}},\
                  {"dec":{"$ne":None}},\
                  {"photoz":{"$ne":None}},\
                  {"sdss_host_offset":{"$ne":"NULL"}},\
                  {"sdss_host_offset":{"$ne":None}}]}
    
    num_results = db.detection_context_features_spec.find(query, projection).count()
    print num_results
    results = db.detection_context_features_spec.find(query, projection)
    #print results
    first_detect_mag["psst"]=[]
    for result in results:
        #first_detect_mag["md"].append(float(result["first_detect_mag"]))
        first_detect_mag["psst"].append(float(result["sdss_host_offset"]))
    num_results = db.detection_context_features_spec.find(query, projection).count()
    results = db.detection_context_features_spec.find(query, projection)
    X_psst, y_psst, ids_psst, surveys = build_data_set(results, num_results)
    print X_psst
    print y_psst
    print ids_psst
    y_psst[y_psst==2] -= 2

    import scipy.io as sio
    sio.savemat("contextual_classification_psst_dataset.mat",{"X":X_psst, "y":y_psst, "ids":ids_psst})

    db = client.detection_context_features_phot
#
    query = {"$and": [{"survey":{"$eq":"md"}},\
                      {"u":{"$ne":"NULL"}},\
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

    num_results = db.detection_context_features_phot.find(query, projection).count()
    results = db.detection_context_features_phot.find(query, projection)
    first_detect_mag["mdp"]=[]
    for result in results:
        first_detect_mag["mdp"].append(float(result["first_detect_mag"]))
    num_results = db.detection_context_features_phot.find(query, projection).count()
    results = db.detection_context_features_phot.find(query, projection)
    X_mdp, y_mdp, ids_mdp, surveys = build_data_set(results, num_results)
    print len(y_mdp)
#
    query = {"$and": [{"survey":{"$eq":"3pi"}},\
                  {"u":{"$ne":"NULL"}},\
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
    
    num_results = db.detection_context_features_phot.find(query, projection).count()
    results = db.detection_context_features_phot.find(query, projection)
    first_detect_mag["3pip"]=[]
    for result in results:
        #first_detect_mag["md"].append(float(result["first_detect_mag"]))
        first_detect_mag["3pip"].append(float(result["sdss_host_offset"]))
    num_results = db.detection_context_features_phot.find(query, projection).count()
    results = db.detection_context_features_phot.find(query, projection)
    X_3pip, y_3pip, ids_3pip, surveys = build_data_set(results, num_results)
    print len(y_3pip)




    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ug_color = X_3pip[:,0] - X_3pip[:,1]
    gr_color = X_3pip[:,1] - X_3pip[:,2]
    delta_detection = X_3pip[:,6]
    #ax.scatter(ug_color[y_3pip==1], gr_color[y_3pip==1], \
    #           delta_detection[y_3pip==1], color="#5E2BFF")
    #ax.scatter(ug_color[y_3pip!=1], gr_color[y_3pip!=1], \
    #           delta_detection[y_3pip!=1], color="#FC6DAB")
    ug_color = X_sdss[:,0] - X_sdss[:,1]
    gr_color = X_sdss[:,1] - X_sdss[:,2]
    delta_detection = X_sdss[:,6]
    #ax.scatter(ug_color[y_sdss==1], gr_color[y_sdss==1], \
    #           delta_detection[y_sdss==1], color="#C04CFD")
    ug_color = X_md[:,0] - X_md[:,1]
    gr_color = X_md[:,1] - X_md[:,2]
    delta_detection = X_md[:,6]
    ax.scatter(ug_color[y_md==1], gr_color[y_md==1], \
               delta_detection[y_md==1], color="#C04CFD")
    ug_color = X_mdp[:,0] - X_mdp[:,1]
    gr_color = X_mdp[:,1] - X_mdp[:,2]
    delta_detection = X_mdp[:,6]
    ax.scatter(ug_color[y_mdp!=1], gr_color[y_mdp!=1], \
               delta_detection[y_mdp!=1], color="#119DA4")
    ax.set_xlabel("u - g")
    ax.set_ylabel("g - r")
    ax.set_zlabel("delta detection")

    plt.show()
    """
    photoX = np.concatenate((X_mdp,X_3pip[y_3pip==0]))
    photoy = np.concatenate((y_mdp,y_3pip[y_3pip==0]))
    #import scipy.io as sio
    #sio.savemat("contextual_classification_photo_dataset.mat",
    #            {"X":photoX, "y":photoy})
    # training set will contain sdss, md, mdp, oldmd
    #train_x = np.concatenate((X_sdss, X_md, X_mdp, X_oldmd, X_3pip))
    #train_x = np.concatenate((X_sdss, X_mdp, X_3pip[y_3pip!=1,:]))
    train_x = X_sdss
    train_ids = ids_sdss
    #train_y = np.concatenate((y_sdss, y_md, y_mdp, y_oldmd, y_3pip))
    #train_y = np.concatenate((y_sdss, y_mdp, y_3pip[y_3pip!=1]))
    train_y = y_sdss
    print "%d training examples" % len(train_y)
    # test set will contain 3pi survey objects
    test_x = np.concatenate((X_3pi, X_fgss, X_oldfgss, X_md, X_oldmd, X_3pip[y_3pip==1]))
    test_y = np.concatenate((y_3pi, y_fgss, y_oldfgss, y_md, y_oldmd, y_3pip[y_3pip==1]))
    test_ids = np.concatenate((ids_3pi, ids_fgss, ids_oldfgss, ids_md, ids_oldmd, ids_3pip[y_3pip==1]))
    print "%d test examples" % len(test_y)


    first_detect_mag = first_detect_mag["sdss"]+first_detect_mag["3pi"]+first_detect_mag["fgss"]+\
                            first_detect_mag["oldfgss"]+ \
                            first_detect_mag["md"]+first_detect_mag["oldmd"]+\
                            list(np.array(first_detect_mag["3pip"])[y_3pip==1])
    print np.shape(first_detect_mag)
    X = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y))
    ids = np.concatenate((train_ids, test_ids))
    m = len(y)
    print m
    np.random.seed(0)
    order = np.random.permutation(m)

    X = X[order]
    y = y[order]
    ids = ids[order]

    first_detect_mag = np.array(first_detect_mag)[order]


    train_x = X[:int(0.75*m),:]
    train_y = y[:int(0.75*m)]
    train_y[np.where(train_y ==2)[0]] -= 2

    test_x = X[int(0.75*m):,:]
    test_y = y[int(0.75*m):]
    test_ids = ids[int(0.75*m):]
    test_y[np.where(test_y ==2)[0]] -= 2
    print np.where(test_y == 0)
    print np.where(test_y == 1)

    first_detect_mag_test = first_detect_mag[int(0.75*m):]

    from sklearn.metrics import f1_score, accuracy_score
    print "All ones benchmark : %.3f" % f1_score(np.squeeze(test_y), np.squeeze(np.ones(np.shape(test_y))))
    print "All zeros benchmark : %.3f" % f1_score(np.squeeze(test_y), np.squeeze(np.zeros(np.shape(test_y))))

    #print first_detect_mag_test
    from sklearn.metrics import f1_score, accuracy_score
    print np.where(test_y != 1.0)
    print "all ones benchmark : %.3f" % f1_score(np.squeeze(test_y), np.squeeze(np.ones(np.shape(test_y))))
    print "all ones benchmark acc: %.3f" % accuracy_score(np.squeeze(test_y), np.squeeze(np.ones(np.shape(test_y))))
    print "All zeros benchmark : %.3f" % f1_score(np.squeeze(test_y), np.squeeze(np.zeros(np.shape(test_y))))
    print "All zeros benchmark acc: %.3f" % accuracy_score(np.squeeze(test_y), np.squeeze(np.zeros(np.shape(test_y))))
    """
    classifier = pickle.load(open("../rf/RF_n_estimators100_max_features4_min_samples_leaf1_contextual_classification_dataset_20150706_shuffled.pkl", "rb"))

#

#    scaler = preprocessing.StandardScaler().fit(train_x)
#    print np.shape(test_x)
#    print np.shape(test_y)
#    print np.shape(np.where(test_y==1)[0])
#    print np.shape(test_x[np.where(test_y==1)[0],:])
#    #test_x = scaler.transform(test_x[np.where(test_y==0)[0],:])
#    test_x = scaler.transform(test_x)
    pred = classifier.predict(test_x)

    font = {"size"   : 20}
    plt.rc("font", **font)
    plt.rc("legend", fontsize=18)
    threshold = 0.5
    #bins = np.arange(15,26,1)
    bins = np.arange(0,5,.25)

    #mags = np.array(first_detect_mag_test)[np.where(test_y==0)[0]]
    mags = np.array(first_detect_mag_test)
    n, bins, patches = plt.hist(mags, bins=bins)
    print n, bins
    bin_allocations = np.digitize(mags, bins)

    MDRs = []
    for i in range(1,len(bins)):
        if n[i-1] == 0:
            MDRs.append(0)
            continue
        preds_for_bin = np.array([pred[np.squeeze(np.where(bin_allocations == i))]])
        y_for_bin = np.array([test_y[np.squeeze(np.where(bin_allocations == i))]])
        print y_for_bin

        #print np.shape(np.where(preds_for_bin != 1))[1] / float(n[i-1])
        #MDRs.append(np.shape(np.where(preds_for_bin != 1))[1] / float(n[i-1]))
        MDRs.append(f1_score(y_for_bin, preds_for_bin))
    print n
    print MDRs
    print np.sum(n[i]*MDRs[i] for i in range(len(n)))/np.sum(n)
    mid_points = []
    for i in range(len(bins)-1):
        mid_points.append(np.mean([bins[i], bins[i+1]]))

    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    #bins = np.arange(15,26,1)
    bins = np.arange(0,5,.25)
    ax1 = ax2.twinx()
    ax2.set_xlabel("Magnitude")
    ax1.set_ylabel("Frequency")
    ax1.set_ylim(ymin=0-0.01*np.max(n), ymax=np.max(n)+0.01*np.max(n))
    n, bins, patches = ax1.hist(mags, bins=bins, color="#3366FF", \
                                alpha=0.25, edgecolor="none")#FF0066

    ax2.set_zorder(ax1.get_zorder()+1)
    ax2.patch.set_visible(False)
    ax2.set_ylim(ymin=-0.01, ymax=1.05)

    print len(MDRs)
    print len(mid_points)
    ax2.plot(mid_points, MDRs, "-",color = "k", lw=3)
    ax2.plot(mid_points, MDRs, "-",label="new", color = "#3366FF", lw=2)
    ax2.plot(mid_points, MDRs, "o", color = "#3366FF", ms=5)#3366FF
    ax2.plot(mid_points+[-1,18,27], 0.978*np.ones(np.shape(mid_points+[-1, 18, 27])), "--", color="k", lw=2)
    
    ax2.set_ylabel("Missed Detection Rate")
    #ax2.set_xlim(xmin=14.9, xmax=25.1)
    ax2.set_xlim(xmin=-0.1, xmax=5.1)
    ax2.grid()
    ax2.text(0.1,0.99,"0.978", size=18, color="k")

    plt.show()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    train_x_pca = pca.fit(train_x).transform(train_x)

    fig = plt.figure()
    """
        
    """
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(train_x_pca[train_y!=1,0],\
               train_x_pca[train_y!=1,1],\
               train_x_pca[train_y!=1,2], color="#FF0066")
               
    ax.scatter(train_x_pca[train_y==1,0],\
               train_x_pca[train_y==1,1],\
               train_x_pca[train_y==1,2], color="#66FF33")
               
    ax.set_xlabel("first eigenvector")
    ax.set_ylabel("second eigenvector")
    ax.set_zlabel("third eigenvector")
    """

    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import f1_score
    import scipy.io as sio
    data = sio.loadmat("../contextual_classification_dataset_pca2_20150707_shuffled.mat")
    scaler = preprocessing.StandardScaler().fit(data["X"])
    clf = GaussianNB()
    train_y = train_y == 1

    #data = sio.loadmat("../contextual_classification_dataset_pca2_20150707_shuffled.mat")
    scaler = preprocessing.StandardScaler().fit(data["X"])
    train_x_pca = scaler.transform(data["X"])
    #train_x_pca = data["X"]
    train_y = np.squeeze(data["y"])
    test_x_pca = scaler.transform(data["testX"])
    #train_x_pca = data["X"]
    test_y = np.squeeze(data["testy"])
    #clf = pickle.load(open("../nn/NerualNet_contextual_classification_dataset_pca2_20150707_shuffled_arch19_lambda0.100000.pkl","rb"))
    clf.fit(train_x_pca, train_y)
    #print clf._architecture
    #pred = np.array(clf.predict_proba(train_x_pca.T) <= .5, dtype="int64")[:,0]
    pred = clf.predict(test_x_pca)
    #print "f1 score : %.3f" % f1_score(np.squeeze(train_y), np.array(clf.predict_proba(train_x_pca.T) <= .5)[:,0])

    h = .02
    x_min, x_max = train_x_pca[:, 0].min() - 1, train_x_pca[:, 0].max() + 2
    y_min, y_max = train_x_pca[:, 1].min() - .1, train_x_pca[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict_proba(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))[:,1]
    Z = Z.reshape(xx.shape)
    print Z

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 10),
             cmap=plt.cm.cool_r, alpha=.7)

    ax1.contour(xx, yy, Z, [0.5], colors='w',lw="10")

    ax1.scatter(test_x_pca[test_y!=1,0],\
               test_x_pca[test_y!=1,1], color="#DC0073", label="not sn",edgecolor="k")

    ax1.scatter(test_x_pca[test_y==1,0],\
           test_x_pca[test_y==1,1], color="#008BF8", label="sn",edgecolor="k")

#    ax1.set_xlabel("first eigenvector")
#    ax1.set_ylabel("second eigenvector")

#    ax1.text(0,-3,"f1 score : %.3f" % f1_score(train_y, pred), color="w")

#    ax1.legend(loc="lower left", numpoints=1)
    plt.show()
    font = {"size"   : 26}
    plt.rc("font", **font)
    plt.rc("legend", fontsize=22)
    #plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    #test_x_pca = pca.transform(test_x)
    test_x_pca = data["testX"]
    test_y = np.squeeze(data["testy"])
    scaler.transform(test_x_pca)
    clf.predict_proba(test_x_pca)
    #pred = np.array(clf.predict_proba(test_x_pca.T) <= .5, dtype="int64")[:,0]
    print pred
    print test_y
    #print "f1 score : %.3f" % f1_score(np.squeeze(test_y), np.array(clf.predict_proba(test_x_pca.T) <= .5)[:,0])
    print test_x[test_y==0][pred[test_y==0]==1]
    print test_ids[test_y==0][pred[test_y==0]==1]
    #test_y = test_y == 1
    import scipy.io as sio
    photo_x_pca = pca.transform(photoX)
    #sio.savemat("contextual_classification_photo_pca2_20150707_shuffled.mat",
    #            {"X":photo_x_pca, "y":photoy})
    ax2 = fig.add_subplot(111)
    
    ax2.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 10),
                cmap=plt.cm.cool_r, alpha=.7)

    ax2.contour(xx, yy, Z, [0.5], colors='w',lw="10")

    ax2.scatter(test_x_pca[test_y==1,0],\
            test_x_pca[test_y==1,1], color="#008BF8",s=50, label="SNe")
    
    ax2.scatter(test_x_pca[test_y!=1,0],\
               test_x_pca[test_y!=1,1], color="#DC0073",s=50,label="not SNe")


    #ax2.scatter(test_x_pca[test_y!=pred,0],\
    #        test_x_pca[test_y!=pred,1], color="none",edgecolor="k", s=100)
    
    ax2.set_xlabel("first principal component")
    ax2.set_ylabel("second principal component")

    #ax2.text(-2,-3,r"$F_1$-score : %.3f" % f1_score(test_y, pred), color="k")
    ax2.set_xlim(x_min,x_max)
    ax2.set_ylim(y_min,y_max)
    print accuracy_score(test_y, pred)
    plt.legend(loc="upper left", numpoints=1)
    plt.show()

    from sklearn.metrics import roc_curve
    pred = clf.predict_proba(train_x_pca)[:,1]
    fpr, tpr, thresholds = roc_curve(train_y, pred)
    plt.plot(1-tpr, fpr)
    pred = clf.predict_proba(test_x_pca)[:,1]
    fpr, tpr, thresholds = roc_curve(test_y, pred)
    plt.plot(1-tpr, fpr)
    plt.show()

    data = sio.loadmat("contextual_classification_dataset_20150706_shuffled.mat")
    x = data["X"]
    scalings = np.max(np.abs(x), axis=0)
    scalings = np.tile(scalings, np.shape(x)[0]).reshape(np.shape(x))
    
    x = np.nan_to_num(x / scalings)
    sigma = np.dot(x, x.transpose()) / (np.shape(x)[1])
    U, S, V = np.linalg.svd(sigma)
    totalVariance = np.sum(S)
    variancesRetained = []
    for i in range(len(S)):
        if np.sum(S[:i])/totalVariance >= 0.99:
            variancesRetained.append(np.sum(S[:i])/totalVariance)
            print "Retaining %d components." % i
            k = i
            break
        variancesRetained.append(np.sum(S[:i])/totalVariance)
    print variancesRetained
    plt.xlabel("Number of Components")
    plt.ylabel("Variance")
    plt.plot(range(len(variancesRetained)), variancesRetained)
    plt.show()

#    import scipy.io as sio
#    sio.savemat("contextual_classification_dataset_20150706_shuffled.mat",
#                {"X":train_x, "y":train_y, "testX":test_x, "testy":test_y})

    bins = [i for i in np.arange(0,6,.1)]
    ax = plt.subplot(111)
    # Only draw spine between the y-ticks
    ax.spines['left'].set_bounds(0, 2.5)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    counts, tmp, patches = ax.hist(train_x[train_y!=1,7], bins=bins, color="k", histtype="step", lw=5)
    ax.hist(train_x[train_y!=1,7], bins=bins, color="#7A8B99", histtype="step", lw=4)
    ax.hist(train_x[train_y==1,7], bins=bins, color="k", histtype="step", lw=5)
    ax.hist(train_x[train_y==1,7], bins=bins, color="#91ADC2", histtype="step", lw=4)
    ax.plot([.5,.5], [0,np.max(counts)+.1*np.max(counts)], "k--", lw=3)
    ax.set_xlabel("offset [arsec]")
    plt.show()
#
    ax = plt.subplot(111)
    # Only draw spine between the y-ticks
    ax.spines['left'].set_bounds(0, 3)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    counts, tmp, patches = ax.hist(test_x[test_y!=1,7], bins=bins, color="k", histtype="step", lw=5)
    ax.hist(test_x[test_y!=1,7], bins=bins, color="#7A8B99", histtype="step", lw=4)
    ax.hist(test_x[test_y==1,7], bins=bins, color="k", histtype="step", lw=5)
    ax.hist(test_x[test_y==1,7], bins=bins, color="#91ADC2", histtype="step", lw=4)
    ax.plot([1,1], [0,np.max(counts)+.1*np.max(counts)], "k--", lw=3)
    ax.set_xlabel("offset [arsec]")
    plt.show()
#
    pred = np.ones(np.shape(train_y))
    pred[train_x[:,7] <= .5] *= 0

    print "training f1 score : %.3f" % f1_score(train_y, pred)

    pred = np.ones(np.shape(test_y))
    pred[test_x[:,7] <= .5] *= 0
    print "test f1 score : %.3f" % f1_score(test_y, pred)
#
#


if __name__ == "__main__":
    main()
