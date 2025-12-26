import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
import pickle
import warnings
from skopt.space import Space
from skopt.sampler import Grid
warnings.filterwarnings("ignore")
from pyomo.opt import SolverFactory
from pyomo.environ import *


filepath_step1_output=""
filepath_result=""

#.............................................add groups(step2).................................................................
df_step1_out=pd.read_excel(fr"{filepath_step1_output}",sheet_name="sheet1",index_col=0,header=0)
df_all_groups=pd.read_excel(r"\data\220group.xlsx",index_col=None,header=0)

df_step2_out=pd.DataFrame(index=df_step1_out.index,columns=df_all_groups.columns)

for i in df_step1_out.index:
    for j in df_step1_out.columns:
        if df_step1_out.loc[i,j]!="NaN":
            df_step2_out.loc[i,j]=df_step1_out.loc[i,j]
df_step2_out=df_step2_out.fillna(0)
df_step2_out.columns=[f"Group {i+1}" for i in range(220)]
df_step2_out.index=[i+1 for i in range(len(df_step1_out.index))]

print(df_step2_out.values)




#...........................................property prediction(step3).............................................................
df_step3_out=pd.DataFrame()
list_property=["tb","tm","hsolp","hf"]
for property in list_property:
    df_par=pd.read_excel(fr"\data\GC par\GC_par_{property}.xlsx",index_col=0)
    df = pd.read_excel(fr"\known dataset\{property}.xlsx",index_col=0)
    train_num = int(1 * len(df.index))
    molecule = [df.loc[j, [f"Group {i + 1}" for i in range(220)]].values for j in range(1, train_num + 1)]


    def JSC(list1, list2):  # index3+index2
        intersection = [min(list1[i], list2[i]) + 1 for i in range(len(list1))]
        union = [max(list1[i], list2[i]) + 1 for i in range(len(list1))]
        res = (np.product(intersection) - 1) / (np.product(union) - 1)
        return res


    def Linear(molecule_new, molecule):
        JSC_dict = {i + 1: JSC(molecule_new, molecule[i]) for i in range(len(molecule))}
        sorted_JSC = {key: JSC_dict[key] for key in sorted(JSC_dict.keys(), key=JSC_dict.get, reverse=True)}
        if list(sorted_JSC.values())[0]==1:
            n=1
        elif list(sorted_JSC.values())[0]>0.9:
            n=5
        elif list(sorted_JSC.values())[0]>0.7:
            n=10
        elif list(sorted_JSC.values())[0]>0.5:
            n=50
        elif list(sorted_JSC.values())[0]>0.3:
            n=200
        else:
            pro_pre=sum([df_par.loc[ig+1,"par"]*molecule_new[ig] for ig in range(220)])+df_par.loc[221,"par"]
            return pro_pre, list(sorted_JSC.values())[0]
        molecule_training = [df.loc[list(sorted_JSC.keys())[i], [f"Group {j + 1}" for j in range(220)]].values for i in range(n)]
        property_training = [df.loc[list(sorted_JSC.keys())[i], f"{property}"] for i in range(n)]

        model = ConcreteModel()
        model.I = Set(initialize=[i for i in range(220)])
        model.par = Var(model.I, within=Reals, initialize=0)
        model.par0 = Var(within=Reals, initialize=0)

        def obj_rule(model):
            error = 0
            for i in range(len(molecule_training)):
                Tc = model.par0
                for j in range(220):
                    Tc += model.par[j] * molecule_training[i][j]
                error += (Tc - property_training[i]) ** 2
            error = (error / len(molecule_training)) ** 0.5
            return error

        model.obj = Objective(rule=obj_rule, sense=minimize)

        opt = SolverFactory('gams')
        io_options = dict()
        io_options['solver'] = "minos"
        io_options['mtype'] = "NLP"
        result = opt.solve(model, tee=False, keepfiles=False, io_options=io_options)

        pro_pre = np.sum([value(model.par[i]) * molecule_new[i] for i in range(220)]) + value(model.par0)
        return pro_pre, list(sorted_JSC.values())[0]

    for i in df_step2_out.index:
        print(i)
        molecule_new = df_step2_out.loc[i, [f"Group {j + 1}" for j in range(220)]].values
        property_pre, JSC_new = Linear(molecule_new, molecule)
        print(property_pre, JSC_new)
        if property=="tb":
            df_step3_out.loc[i, f"{property}_pre"] = np.log(property_pre)*244.5165
        elif property=="tm":
            df_step3_out.loc[i, f"{property}_pre"] = np.log(property_pre)*143.5706
        else:
            df_step3_out.loc[i, f"{property}_pre"] = property_pre
        df_step3_out.loc[i, f"{property}_JSC"] = JSC_new









#.......................................screen and check molecule(step4)..........................................................
df_step5_out=df_step3_out

for i in df_step5_out.index:
    if (df_step5_out.loc[i,"tb_pre"]<430 or df_step5_out.loc[i,"tm_pre"]>270 or df_step5_out.loc[i,"hsolp_pre"]<20):
        df_step5_out=df_step5_out.drop(index=i)

list_property=["tb","tm","hsolp","hf"]
for property in list_property:
    df_property=pd.read_excel(fr"\data\knwon dataset\{property}.xlsx",index_col=0)
    for i in df_step5_out.index:
        print(i)
        for jj in df_property.index:
            if df_property.loc[jj,[f"Group {j + 1}" for j in range(220)]].values.tolist()==df_step2_out.loc[i,[f"Group {j + 1}" for j in range(220)]].values.tolist():
                if property=="tb":
                    df_step5_out.loc[i,f"{property}_real"]=np.log(df_property.loc[jj,f"{property}"])*244.5165
                elif property=="tm":
                    df_step5_out.loc[i,f"{property}_real"]=np.log(df_property.loc[jj,f"{property}"])*143.5706
                else:
                    df_step5_out.loc[i,f"{property}_real"]=df_property.loc[jj,f"{property}"]

df_step5_out.to_excel(fr"{filepath_result}")








