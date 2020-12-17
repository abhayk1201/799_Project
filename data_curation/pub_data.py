import wget
import os
pmid_list = open("pubmed_result.txt").readlines()
pmid_list1 = pmid_list[10000:20000]

empty_files  = open("empty_files1.txt","a")
good_pmid =  open("good_pmid1.txt","a")
i= 0
for pmid in pmid_list1:
    pmid = pmid.strip("\n")
    print(pmid)
    url = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator?pmids="+pmid
    if os.path.exists("temp.txt"):
        os.remove("temp.txt") 
    wget.download(url, "temp.txt") 
    if not os.stat("temp.txt").st_size == 0:
        good_pmid.write(str(pmid) + "\n")
        wget.download(url, pmid+".txt") 
        
    else:
        empty_files.write(str(pmid) + "\n")
    print("\n"+ str(i))
    i += 1
empty_files.close()
good_pmid.close()        