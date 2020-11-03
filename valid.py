import torch
def calculate_class_wise_precision_recall_f1(test_predictions,test_labels,classes):
    test_predictions = tuple(test_predictions)
    test_labels = tuple(test_labels)
    test_predictions = torch.cat(test_predictions,0)
    test_labels = torch.cat(test_labels,0)
    stat_dict = {}
    for i,class_name in enumerate(classes):
        stat_dict[class_name] = {}
        stat_dict[class_name]['tp'] = test_predictions[(test_labels == i).nonzero()].eq(i).nonzero().size(0)
        stat_dict[class_name]['fp'] = test_predictions[(test_labels != i).nonzero()].eq(i).nonzero().size(0)
        stat_dict[class_name]['fn'] = test_predictions[(test_labels == i).nonzero()].ne(i).nonzero().size(0)
        stat_dict[class_name]['precision'] = round(stat_dict[class_name]['tp']/(stat_dict[class_name]['tp'] + stat_dict[class_name]['fp']+0.0001),2)
        stat_dict[class_name]['recall'] = round(stat_dict[class_name]['tp']/(stat_dict[class_name]['tp'] + stat_dict[class_name]['fn']+ 0.0001),2)
        stat_dict[class_name]['f1'] = round(2 * stat_dict[class_name]['precision'] * stat_dict[class_name]['recall']/(stat_dict[class_name]['precision'] + stat_dict[class_name]['recall']+0.0001),2)

    stat_dict['accuracy'] = round(test_labels.eq(test_predictions).nonzero().size(0)/test_labels.size(0),2)

    return stat_dict







