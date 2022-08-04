import requests

def get_url(url,port):
    #try:
    #    ip = requests.get("http://169.254.169.254/latest/meta-data/public-ipv4").text
    #    ip = "http://" + ip  + ":" + str(port) + "/" + url
    #except:
    ip = "http://localhost:"+ str(port) + "/" + url

    return ip


def format_stats(stats,classes):
    formatted_stats =[]
    for key,value in stats.items():
        formatted_stats.append({"class_name":classes[key],"precision":value['p'],"recall":value['r'],"f1":value["f1"]})
    metrics = ["Precision","Recall","F1"]
    return formatted_stats,metrics

def format_file_stats(file_stats):
    formatted_file_stats = {}
    fns = {}
    fps = {}
    tps = {}
    for class_id,file_stat in file_stats.items():
        fns[class_id] = file_stat["fn"]
        fps[class_id] = file_stat['fp']
        tps[class_id] = file_stat['tp']

    formatted_file_stats['fn'] = fns
    formatted_file_stats['fp'] = fps
    formatted_file_stats['tp'] = tps

    return formatted_file_stats


