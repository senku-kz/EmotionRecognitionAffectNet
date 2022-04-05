from tabulate import tabulate


def save_value(filename, value):
    print(f'Total model accuracy {value}%')
    with open(filename, 'w') as fp_file:
        fp_file.write(f'Total model accuracy {value}%')


def save_dic_as_table(filename, v_dic):
    tbl_lst = []
    headers = ["Emotions", "Predictions accuracy (%)"]
    for k, v in sorted(v_dic.items()):
        tbl_lst.append((k, v,))
    print(tabulate(tbl_lst, headers=headers))
    with open(filename, 'w') as fp_file:
        fp_file.write(tabulate(tbl_lst, headers=headers))


def save_dic_as_table(filename, v_dic_1, v_dic_2, v_dic_3):
    tbl_lst = []
    headers = ["Emotions", "Correct predictions", "Total predictions", "Predictions accuracy (%)"]
    for k, v in sorted(v_dic_1.items()):
        tbl_lst.append((k, v_dic_1[k], v_dic_2[k], v_dic_3[k]))
    print(tabulate(tbl_lst, headers=headers))
    with open(filename, 'w') as fp_file:
        fp_file.write(tabulate(tbl_lst, headers=headers))


if __name__ == '__main__':
    d = {"Dave": ("13", "Male"), "Sarah": ("16", "Female")}

    headers = ["Name", "Age", "Gender"]
    print([(k,) + v for k, v in d.items()])
    print(tabulate([(k,) + v for k, v in d.items()], headers=headers))
