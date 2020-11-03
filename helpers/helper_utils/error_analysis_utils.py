import requests

def get_url(url,port):
    try:
        ip = requests.get("http://169.254.169.254/latest/meta-data/public-ipv4").text
        ip = "http://" + ip  + ":" + str(port) + "/" + url
    except:
        ip = "http://localhost:"+ str(port) + "/" + url

    return ip


def format_stats(stats,classes):
    formatted_stats =[]
    for key,value in stats.items():
        formatted_stats.append({"class_name":classes[key],"precision":value['p'],"recall":value['r'],"f1":value["f1"]})
    metrics = ["Precision","Recall","F1"]
    return formatted_stats,metrics
