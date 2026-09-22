import re
import argparse
from pprint import pprint
import os
import sys

def analyze_alignment(align_folder, out_file=None):
    root = align_folder
    if not root.endswith(os.sep):
        root += os.sep
    
    # Use os.path.join for robust path handling
    ref_human_path = os.path.join(align_folder, "ref_human_detail")
    human_our_path = os.path.join(align_folder, "human_our_detail")
    ref_our_path = os.path.join(align_folder, "ref_our_detail")

    dic={}
    insert = 0 
    delete = 0
    sub = 0
    cor=0
    count=0
    
    # Check if files exist
    if not os.path.exists(ref_human_path):
        return {"Error": f"File not found: {ref_human_path}"}

    with open(ref_human_path,'r', encoding='utf-8') as f:
        ##  0： ref  1：human 2：ops --- 3: human  4： our  5: ops 
        for line in f:
            line = line.strip()
            if("ref" in line ):
                ref = line.split("ref")
                ref[0] = ref[0].strip(" ")
                ref[1] = ref[1].strip(" ")
                ref[1] = re.sub(" +"," ",ref[1])
                ref_seq = ref[1].split(" ")
                dic[ref[0]] = []
                dic[ref[0]].append(ref[1])
            elif( "hyp" in line ):
                hyp = line.split("hyp")
                hyp[0] = hyp[0].strip(" ")
                hyp[1] = hyp[1].strip(" ")
                hyp[1] = re.sub(" +"," ",hyp[1])
                hyp_seq = hyp[1].split(" ")
                dic[hyp[0]].append(hyp[1])
            elif( " op " in line ):   
                op = line.split(" op ")
                op[0] = op[0].strip(" ")
                op[1] = op[1].strip(" ")
                op[1] = re.sub(" +"," ",op[1])
                op_seq = op[1].split(" ")
                dic[op[0]].append(op[1])
                for i in op_seq:
                    if(i == "I"):
                        insert+=1
                    elif(i == "D"):
                        delete+=1
                        count+=1
                    elif(i == "S"):
                        sub +=1
                        count+=1
                    elif(i=="C"):
                        cor+=1
                        count+=1
    
    if not os.path.exists(human_our_path):
        return {"Error": f"File not found: {human_our_path}"}

    with open(human_our_path,'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split(" ")
            if not parts: continue
            fn = parts[0]
            if(fn not in dic):
                continue
            if("ref" in line ):
                ref = line.split("ref")
                ref[0] = ref[0].strip(" ")
                ref[1] = ref[1].strip(" ")
                ref[1] = re.sub(" +"," ",ref[1])
                ref_seq = ref[1].split(" ")
                dic[ref[0]].append(ref[1])
            elif( "hyp" in line ):
                hyp = line.split("hyp")
                hyp[0] = hyp[0].strip(" ")
                hyp[1] = hyp[1].strip(" ")
                hyp[1] = re.sub(" +"," ",hyp[1])
                hyp_seq = hyp[1].split(" ")
                dic[hyp[0]].append(hyp[1])
            elif( " op " in line ):
                op = line.split(" op ")
                op[0] = op[0].strip(" ")
                op[1] = op[1].strip(" ")
                op[1] = re.sub(" +"," ",op[1])
                op_seq = op[1].split(" ")
                dic[op[0]].append(op[1])

    if not os.path.exists(ref_our_path):
         return {"Error": f"File not found: {ref_our_path}"}

    with open(ref_our_path,'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split(" ")
            if not parts: continue
            fn = parts[0]
            if(fn not in dic):
                continue
            if("ref" in line ):
                ref = line.split("ref")
                ref[0] = ref[0].strip(" ")
                ref[1] = ref[1].strip(" ")
                ref[1] = re.sub(" +"," ",ref[1])
                ref_seq = ref[1].split(" ")
                dic[ref[0]].append(ref[1])
            elif( "hyp" in line ):
                hyp = line.split("hyp")
                hyp[0] = hyp[0].strip(" ")
                hyp[1] = hyp[1].strip(" ")
                hyp[1] = re.sub(" +"," ",hyp[1])
                hyp_seq = hyp[1].split(" ")
                dic[hyp[0]].append(hyp[1])
            elif( " op " in line ):
                op = line.split(" op ")
                op[0] = op[0].strip(" ")
                op[1] = op[1].strip(" ")
                op[1] = re.sub(" +"," ",op[1])
                op_seq = op[1].split(" ")
                dic[op[0]].append(op[1])

    cor_cor = 0
    cor_cor1 = 0 
    cor_nocor = 0
    
    sub_sub = 0
    sub_sub1 = 0
    sub_nosub = 0
    
    ins_ins = 0
    ins_ins1 = 0
    ins_noins =0
    
    del_del = 0
    del_del1 = 0
    del_nodel =0
    
    y_true = []
    y_pred = []
    yy_true = []
    yy_pred = []
    # print(hyp)
    
    for i in dic:
        arr = dic[i]
        # Safety check for array length
        if len(arr) < 9:
            continue

        # del detection 
        TTTTTRRRRR = 0
        wav_id = i
        # print(arr)
        ref_seq = arr[0].split(" ")
        ref_seq3 = arr[6].split(" ")
        our_seq3 = arr[7].split(" ")
        op =  arr[2].split(" ")
        op3 = arr[8].split(" ")
        
        # break
        flag = 0
        for i in range( len(ref_seq) ):
            if(ref_seq[i] == "<eps>"):
                continue
            while(flag < len(ref_seq3) and ref_seq3[flag] == "<eps>"):
                flag+=1  
            if flag < len(ref_seq3) and ( ref_seq[i]  == ref_seq3[flag] and ref_seq[i]!="<eps>" ):
                if( op[i] == "D"  and op3[flag] == "D" ):
                    del_del+=1
                    yy_true.append(ref_seq[i])
                    yy_pred.append('eer')
                elif( op[i] == "D" and op3[flag] != "D" and op3[flag] != "C"):
                    del_del1+=1
                    pho = ref_seq[i]
                    yy_true.append(ref_seq[i])
                    yy_pred.append(our_seq3[flag])
                    debug = 1
                elif( op[i] == "D" and op3[flag] != "D" and op3[flag] == "C"):
                    del_nodel+=1
                    pho = ref_seq[i]
                    yy_true.append(ref_seq[i])
                    yy_pred.append(ref_seq[i])
                    debug = 1
                flag+=1  
                
    
        ## cor ins sub detection 
        ref_seq = arr[0].split(" ")
        human_seq = arr[1].split(" ")
        op =  arr[2].split(" ")
        human_seq2 = arr[3].split(" ")
        our_seq2 = arr[4].split(" ")
        op2 = arr[5].split(" ")
    
        flag = 0 
        for i in range( len(human_seq) ):
            if(human_seq[i] == "<eps>"):
                continue
            while(human_seq2[flag] == "<eps>"):
                flag+=1
            if( human_seq[i]  == human_seq2[flag] and human_seq[i]!="<eps>" ):
                if( op[i] == "C"  and op2[flag] == "C" ):
                    cor_cor+=1
                    y_true.append(human_seq[i])
                    y_pred.append(human_seq[i])
                elif( op[i] == "C" and op2[flag] != "C"):
                    cor_nocor+=1
                    y_true.append(human_seq[i])
                    y_pred.append(our_seq2[flag])
    
    
                if( op[i] == "S" and op2[flag] == "C" ):
                    sub_sub+=1
                    pho = ref_seq[i]
                    yy_true.append(ref_seq[i])
                    yy_pred.append(human_seq2[flag])
                    debug = 1
                elif( op[i] == "S"  and op2[flag] !="C" and ref_seq[i] != our_seq2[flag]):
                    sub_sub1+=1
                    pho = ref_seq[i]
                    yy_true.append(ref_seq[i])
                    yy_pred.append(our_seq2[flag])
                    debug = 1
    
                elif( op[i] == "S"  and op2[flag] !="C" and ref_seq[i] == our_seq2[flag]):
                    sub_nosub+=1
                    pho = ref_seq[i]
                    yy_true.append(ref_seq[i])
                    yy_pred.append(ref_seq[i])
                    debug = 1
                
                if(op[i] == "I" and op2[flag] == "C" ):
                    ins_ins+=1
    
                elif( op[i] == "I" and op2[flag]!="C" and op2[flag]!="D"):
                    ins_ins1+=1
                elif( op[i] == "I" and op2[flag]!="C" and op2[flag]=="D"):
                    ins_noins+=1
    
                flag+=1
    
    sum1 = cor_cor + cor_nocor + sub_sub + sub_sub1 + sub_nosub + ins_ins + ins_ins1 + ins_noins + del_del + del_del1 + del_nodel  
    # print("sum:",sum1)
    TR = sub_sub + sub_sub1  +  +del_del1+del_del + ins_ins1 +  ins_ins
    FR = cor_nocor # 
    FA = sub_nosub + ins_noins + del_nodel 
    TA = cor_cor # 
    
    recall = TR/(TR+FA) if (TR+FA) > 0 else 0
    precision = TR/(TR+FR) if (TR+FR) > 0 else 0
    f1_score = 2*precision*recall/(recall+precision) if (recall+precision) > 0.0 else 0.0
    
    true_accept_rate = cor_cor/(cor_cor+cor_nocor) if (cor_cor+cor_nocor) > 0 else 0
    false_reject_rate = cor_nocor/(cor_cor+cor_nocor) if (cor_cor+cor_nocor) > 0 else 0
    
    err_count = sub_sub+sub_sub1+sub_nosub+ins_ins+ins_ins1+ins_noins+del_del+del_del1+del_nodel
    false_accept = sub_nosub + ins_noins + del_nodel
    
    Correct_Diag = sub_sub + ins_ins + del_del
    Error_Diag =  sub_sub1 + ins_ins1 + del_del1
    
    false_accept_ratio = false_accept/err_count if err_count > 0 else 0
    correct_diag_ratio = Correct_Diag/(Correct_Diag+Error_Diag) if (Correct_Diag+Error_Diag) > 0 else 0
    error_diag_ratio = Error_Diag/(Correct_Diag+Error_Diag) if (Correct_Diag+Error_Diag) > 0 else 0
    
    FAR = 1-recall
    FRR = cor_nocor/(cor_nocor+cor_cor) if (cor_nocor+cor_cor) > 0 else 0
    DER = Error_Diag / (Error_Diag + Correct_Diag) if (Error_Diag + Correct_Diag) > 0 else 0
    detection_acc = (TA+TR)/(TR+TA+FR+FA) if (TR+TA+FR+FA) > 0 else 0
    
    metrics = {
        "Recall": recall,
        "Precision": precision,
        "F1-score": f1_score,
        "TA": true_accept_rate,
        "FR": false_reject_rate,
        "FA": false_accept_ratio,
        "CD": correct_diag_ratio,
        "ED": error_diag_ratio,
        "FAR": FAR,
        "FRR": FRR,
        "DER": DER,
        "DetAcc": detection_acc,
        "PER": 1.0 - detection_acc 
    }
    
    # Write to file if specified
    if out_file:
         with open(out_file, 'a') as f:
            print("Recall: %.4f" %(recall), file=f)
            print("Precision: %.4f" %(precision), file=f)
            print("f1-score: %.4f" % (f1_score), file=f)
            
            print("TA: %.4f" %(true_accept_rate), file=f)
            print("FR: %.4f" %(false_reject_rate), file=f)
            
            print("FA: %.4f" %(false_accept_ratio), file=f)
            print("CD: %.4f" %(correct_diag_ratio), file=f)
            print("ED: %.4f" %(error_diag_ratio), file=f)
            
            print("FAR: %.4f" %(FAR), file=f)
            print("FRR: %.4f" %(FRR), file=f)
            print("DER: %.4f" %(DER), file=f)
            print("DetAcc: %.4f" % (detection_acc), file=f)
            print('\n', file=f)

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate and align predicted phonemes with ground truth.")

    parser.add_argument(
        '--out-file',
        type=str,
        default=None,
        help="Optional path to output metrics file. "
    )

    parser.add_argument(
        '--align-folder',
        type=str,
        default='aligned',
        help="Directory of alignment detail files (default: 'aligned')."
    )

    args = parser.parse_args()
    
    metrics = analyze_alignment(args.align_folder, args.out_file)
    if not args.out_file:
        pprint(metrics)