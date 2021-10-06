import torch
from utils.dash2mat import dash2matrix

# test_seq = "AAGGGUUAUGCCCUUUAAG"
# out_label_dash = ".(.))((.)"
# out_label = torch.zeros((9,3))
# out_label[:,0] = torch.tensor([0.8,0.2,0.6,0.05,0.2,0.23,0.15,0.7,0.2])
# out_label[:,1] = torch.tensor([0.1,0.5,0.3,0.05,0.4,0.7,0.8,0.4,0.3])
# out_label[:,2] = torch.tensor([0.2,0.2,0.3,0.75,0.8,0.2,0.3,0.1,0.6])


def adjust_algorithm(out):
    # put in a out_label B,L,3 -> B, L, L
    check_stack = []
    # margin = []  # keep all the margins of '('
    # num_of_margin = 0
    final_pred = []
    # flag = True
    batch_size = out.shape[0]
    batch_final_matrix = torch.zeros(batch_size, out.shape[1], out.shape[1])
    for j in range(batch_size):
        final_pred.append([])
        for i in range(out.shape[1]):
            if(torch.argmax(out[j][i])==0):
                final_pred[j].append('.')
                # flag = True
            elif(torch.argmax(out[j][i])==1):
                check_stack.append(i)
                final_pred[j].append('(')
                # if(flag):
                #     margin.append([i])
                #     num_of_margin += 1
                #     flag = False
                # if((flag==False) and torch.argmax(out[i+1])!=1):
                #     print(type(margin[num_of_margin-1]))
                #     margin[num_of_margin-1].append(i)
            else:
                if(len(check_stack)):
                    check_stack.pop()
                    final_pred[j].append(')')
                else:
                    final_pred[j].append('.')
                # flag = True
        if(len(check_stack)):
            for i in range(len(check_stack)):
                final_pred[j][check_stack[i]]='.'
            check_stack = []

        batch_final_matrix[j] = dash2matrix(''.join(final_pred[j]),out.shape[1])
    return batch_final_matrix

# out_label = out_label.unsqueeze(0).repeat(4,1,1)
# # print(out_label.shape)
#
# result = adjust_algorithm(out_label)
# print(result.shape)

