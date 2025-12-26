import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
import pickle
import warnings
from skopt.space import Space
from skopt.sampler import Grid
from ProductDesignFunction import RMSE,index_of_train, index_of_test, GP_tuning, RMSE, input_modi, error_analysis, \
    validation_loss, find_min_index, GP, plot_quartile, hyper_tuning
warnings.filterwarnings("ignore")
from pyomo.opt import SolverFactory
from pyomo.environ import *




filepath_step1_output=""
filepath_result=""
filepath_input_process=""

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
df_targetmol=df_step2_out
df_step3_out=pd.DataFrame()
#.........................................predict tb, tc, pc................................................................

def fun_wholemodel(train_input,train_output,test_input,test_output):
    # Linear prior...............................
    print("-----------------simple1----------------")
    svr_lin = SVR(kernel="linear", C=100, gamma="auto", epsilon=0.4)
    model_simple = svr_lin.fit(train_input, train_output)
    print("-----------------stop training------------------")
    test_predict_simple = model_simple.predict(test_input)
    rmse_test_simple = RMSE(test_predict_simple, test_output)
    print("rmse_test_simple:", rmse_test_simple)
    train_predict_simple = model_simple.predict(train_input)
    rmse_train_simple = RMSE(train_predict_simple, train_output)
    print("rmse_train_simple:", rmse_train_simple)

    # GP model...............................
    fold_number = 5
    alpha_num = 5
    grid_number = 100

    test_output_linear = test_predict_simple
    test_output_nonlinear = test_output - test_output_linear
    train_output_linear = train_predict_simple
    train_output_nonlinear = train_output - train_output_linear

    space = Space([(0.1, 1), (5.0, 10.0), (0.1, 10.0), (10.0, 100.0)])
    grid = Grid(border="include", use_full_layout=False)  # 栅格采样
    hyperparameter = grid.generate(space.dimensions, grid_number)
    alfaset = np.linspace(1.2, 4, alpha_num)

    #start to train GP.........................
    loss_record = []
    l_record = []
    for i2 in range(len(alfaset)):
        alfa = alfaset[i2]
        print(f"--------------alfa:{alfa}----------------")
        l = np.ones(len(hyperparameter[0])) * 10
        test_input_distort = np.log(test_input + 1) / np.log(alfa)
        train_input_distort = np.log(train_input + 1) / np.log(alfa)
        sigma_e = 1e-5
        hyper = hyper_tuning(train_input_distort, train_output_nonlinear, grid_number, fold_number)
        l = hyper['length_scale']
        l_record.append(l)
        loss_record.append(hyper["minimum loss"])
    index_ = find_min_index(np.array(loss_record))[1]
    l = l_record[index_]
    alfa = alfaset[index_]

    #prediction..................................
    test_input_distort = np.log(test_input + 1) / np.log(alfa)
    train_input_distort = np.log(train_input + 1) / np.log(alfa)
    test_predict =GP(train_input_distort, train_output_nonlinear, test_input_distort, test_output_nonlinear, l, 1e-5, 1)[1]
    rmse_test = RMSE(test_output, test_predict + test_output_linear)
    print("rmse_modi_test:", rmse_test)
    train_predict =GP(train_input_distort, train_output_nonlinear, train_input_distort, train_output_nonlinear, l, 1e-5, 1)[1]
    rmse_train = RMSE(train_output, train_predict + train_output_linear)
    print("rmse_modi_train:", rmse_train)

    save = {"length_scale": l, "sigma_e": 1e-5, "alfa": alfa,
            "test_predict": test_predict + test_output_linear,
            "train_predict": train_predict + train_output_linear, }
    return train_predict + train_output_linear,test_predict + test_output_linear




def fun_wholemodel_tc(train_input,train_output,test_input,test_output):
    # Linear prior...............................
    print("-----------------simple1----------------")
    svr_lin = SVR(kernel="linear", C=100, gamma="auto", epsilon=0.4)
    model_simple = svr_lin.fit(train_input, train_output)
    print("-----------------stop training------------------")
    test_predict_simple = model_simple.predict(test_input)
    rmse_test_simple = RMSE(test_predict_simple, test_output)
    print("rmse_test_simple:", rmse_test_simple)
    train_predict_simple = model_simple.predict(train_input)
    rmse_train_simple = RMSE(train_predict_simple, train_output)
    print("rmse_train_simple:", rmse_train_simple)

    # GP model...............................

    test_output_linear = test_predict_simple
    test_output_nonlinear = test_output - test_output_linear
    train_output_linear = train_predict_simple
    train_output_nonlinear = train_output - train_output_linear



    l = [0.4, 5.0, 0.1, 30.0]
    alfa = 4.0

    #prediction..................................
    test_input_distort = np.log(test_input + 1) / np.log(alfa)
    train_input_distort = np.log(train_input + 1) / np.log(alfa)
    test_predict =GP(train_input_distort, train_output_nonlinear, test_input_distort, test_output_nonlinear, l, 1e-5, 1)[1]
    rmse_test = RMSE(test_output, test_predict + test_output_linear)
    print("rmse_modi_test:", rmse_test)
    train_predict =GP(train_input_distort, train_output_nonlinear, train_input_distort, train_output_nonlinear, l, 1e-5, 1)[1]
    rmse_train = RMSE(train_output, train_predict + train_output_linear)
    print("rmse_modi_train:", rmse_train)

    save = {"length_scale": l, "sigma_e": 1e-5, "alfa": alfa,
            "test_predict": test_predict + test_output_linear,
            "train_predict": train_predict + train_output_linear, }
    return train_predict + train_output_linear,test_predict + test_output_linear





def fun_wholemodel_tb(train_input,train_output,test_input,test_output):
    # Linear prior...............................
    print("-----------------simple1----------------")
    svr_lin = SVR(kernel="linear", C=100, gamma="auto", epsilon=0.4)
    model_simple = svr_lin.fit(train_input, train_output)
    print("-----------------stop training------------------")
    test_predict_simple = model_simple.predict(test_input)
    rmse_test_simple = RMSE(test_predict_simple, test_output)
    print("rmse_test_simple:", rmse_test_simple)
    train_predict_simple = model_simple.predict(train_input)
    rmse_train_simple = RMSE(train_predict_simple, train_output)
    print("rmse_train_simple:", rmse_train_simple)

    # GP model...............................

    test_output_linear = test_predict_simple
    test_output_nonlinear = test_output - test_output_linear
    train_output_linear = train_predict_simple
    train_output_nonlinear = train_output - train_output_linear


    l = [1.0, 5.0, 0.1, 50.0]
    alfa = 1.9

    #prediction..................................
    test_input_distort = np.log(test_input + 1) / np.log(alfa)
    train_input_distort = np.log(train_input + 1) / np.log(alfa)
    test_predict =GP(train_input_distort, train_output_nonlinear, test_input_distort, test_output_nonlinear, l, 1e-5, 1)[1]
    rmse_test = RMSE(test_output, test_predict + test_output_linear)
    print("rmse_modi_test:", rmse_test)
    train_predict =GP(train_input_distort, train_output_nonlinear, train_input_distort, train_output_nonlinear, l, 1e-5, 1)[1]
    rmse_train = RMSE(train_output, train_predict + train_output_linear)
    print("rmse_modi_train:", rmse_train)

    save = {"length_scale": l, "sigma_e": 1e-5, "alfa": alfa,
            "test_predict": test_predict + test_output_linear,
            "train_predict": train_predict + train_output_linear, }
    return train_predict + train_output_linear,test_predict + test_output_linear






def fun_wholemodel_pc(train_input,train_output,test_input,test_output):
    # Linear prior...............................
    print("-----------------simple1----------------")
    svr_lin = SVR(kernel="linear", C=100, gamma="auto", epsilon=0.4)
    model_simple = svr_lin.fit(train_input, train_output)
    print("-----------------stop training------------------")
    test_predict_simple = model_simple.predict(test_input)
    rmse_test_simple = RMSE(test_predict_simple, test_output)
    print("rmse_test_simple:", rmse_test_simple)
    train_predict_simple = model_simple.predict(train_input)
    rmse_train_simple = RMSE(train_predict_simple, train_output)
    print("rmse_train_simple:", rmse_train_simple)

    # GP model...............................

    test_output_linear = test_predict_simple
    test_output_nonlinear = test_output - test_output_linear
    train_output_linear = train_predict_simple
    train_output_nonlinear = train_output - train_output_linear



    l = [1.0, 5.0, 3.4, 20.0]
    alfa = 1.2

    #prediction..................................
    test_input_distort = np.log(test_input + 1) / np.log(alfa)
    train_input_distort = np.log(train_input + 1) / np.log(alfa)
    test_predict =GP(train_input_distort, train_output_nonlinear, test_input_distort, test_output_nonlinear, l, 1e-5, 1)[1]
    rmse_test = RMSE(test_output, test_predict + test_output_linear)
    print("rmse_modi_test:", rmse_test)
    train_predict =GP(train_input_distort, train_output_nonlinear, train_input_distort, train_output_nonlinear, l, 1e-5, 1)[1]
    rmse_train = RMSE(train_output, train_predict + train_output_linear)
    print("rmse_modi_train:", rmse_train)

    save = {"length_scale": l, "sigma_e": 1e-5, "alfa": alfa,
            "test_predict": test_predict + test_output_linear,
            "train_predict": train_predict + train_output_linear, }
    return train_predict + train_output_linear,test_predict + test_output_linear









property_list=["tb","tc","pc"]
for property in property_list:
    df_candidatetraining_GP=pd.read_excel(fr"\data\known dataset\{property}.xlsx",index_col=0).head(1000)
    molecule=[df_candidatetraining_GP.loc[j, [f"Group {i + 1}" for i in range(220)]].values for j in range(1,len(df_candidatetraining_GP.index)+1)]

    def JSC(list1,list2):     #index3+index2
        intersection = [min(list1[i],list2[i])+1 for i in range(len(list1))]
        union = [max(list1[i],list2[i])+1 for i in range(len(list1))]
        res=(np.product(intersection)-1)/(np.product(union)-1)
        return res

    def Linear(molecule_new,molecule):
        JSC_dict={i+1:JSC(molecule_new,molecule[i]) for i in range(len(molecule))}
        sorted_JSC = {key: JSC_dict[key] for key in sorted(JSC_dict.keys(), key=JSC_dict.get,  reverse=True)}

        if list(sorted_JSC.values())[0] ==1:
            return df_candidatetraining_GP.loc[list(sorted_JSC.keys())[0], f"{property}"],1
        elif list(sorted_JSC.values())[0] > 0.9:
            n = 10
        elif list(sorted_JSC.values())[0] > 0.7:
            n = 20
        elif list(sorted_JSC.values())[0] > 0.5:
            n = 50
        elif list(sorted_JSC.values())[0] > 0.2:
            n = 100
        else:
            molecule_training = [df_candidatetraining_GP.loc[j, [f"Group {i + 1}" for i in range(220)]].values for j in range(1, len(df_candidatetraining_GP.index) + 1)]
            property_training = [df_candidatetraining_GP.loc[j, f"{property}"] for j in range(1, len(df_candidatetraining_GP.index) + 1)]
            if property == "tc":
                _, pro_pre = fun_wholemodel_tc(np.array(molecule_training).astype(float),np.array(property_training).astype(float),np.array([molecule_new]).astype(float),np.array([0]).astype(float))
            if property == "tb":
                _, pro_pre = fun_wholemodel_tb(np.array(molecule_training).astype(float),np.array(property_training).astype(float),np.array([molecule_new]).astype(float),np.array([0]).astype(float))
            if property == "pc":
                _, pro_pre = fun_wholemodel_pc(np.array(molecule_training).astype(float),np.array(property_training).astype(float),np.array([molecule_new]).astype(float),np.array([0]).astype(float))

            return pro_pre[0], list(sorted_JSC.values())[0]
        print(n)
        # n=5

        molecule_training=[df_candidatetraining_GP.loc[list(sorted_JSC.keys())[i], [f"Group {j + 1}" for j in range(220)]].values for i in range(n)]
        property_training=[df_candidatetraining_GP.loc[list(sorted_JSC.keys())[i], f"{property}"] for i in range(n)]
        _,pro_pre=fun_wholemodel(np.array(molecule_training).astype(float),np.array(property_training).astype(float),np.array([molecule_new]).astype(float),np.array([0]).astype(float))
        return pro_pre[0],list(sorted_JSC.values())[0]


    for i in range(1,len(df_targetmol.index)+1):
        print(i)
        molecule_new = df_targetmol.loc[i, [f"Group {j + 1}" for j in range(220)]].values
        property_pre,JSC_new=Linear(molecule_new, molecule)
        print(property_pre,JSC_new)
        df_step3_out.loc[i, f"{property}_pre"] = property_pre
        df_step3_out.loc[i, f"JSC_{property}"] = JSC_new






#.........................................predict cpg................................................................
df_candidatetraining_cpg=pd.read_excel(r"\data\known dataset\cpg.xlsx",index_col=0)
molecule=[df_candidatetraining_cpg.loc[j, [f"Group {i + 1}" for i in range(220)]].values for j in df_candidatetraining_cpg.index]
df_GC_par = pd.read_excel(fr"\data\GC par\GC_par_cpg.xlsx",index_col=0)

def JSC(list1,list2):     #index3+index2
    intersection = [min(list1[i],list2[i])+1 for i in range(len(list1))]
    union = [max(list1[i],list2[i])+1 for i in range(len(list1))]
    res=(np.product(intersection)-1)/(np.product(union)-1)
    return res

def Linear(molecule_new,molecule):

    JSC_dict={i+1:JSC(molecule_new,molecule[i]) for i in range(len(molecule))}
    sorted_JSC = {key: JSC_dict[key] for key in sorted(JSC_dict.keys(), key=JSC_dict.get,  reverse=True)}

    if list(sorted_JSC.values())[0] == 1:
        n = 1
    elif list(sorted_JSC.values())[0] > 0.9:
        n = 10
    elif list(sorted_JSC.values())[0] > 0.7:
        n = 20
    elif list(sorted_JSC.values())[0] > 0.5:
        n = 50
    elif list(sorted_JSC.values())[0] > 0.2:
        n = 100
    else:
        A_pre = np.sum([df_GC_par.loc[i + 1, "par_A"] * molecule_new[i] for i in range(220)]) + df_GC_par.loc[221, "par_A"]
        B_pre = np.sum([df_GC_par.loc[i + 1, "par_B"] * molecule_new[i] for i in range(220)]) + df_GC_par.loc[221, "par_B"]
        C_pre = np.sum([df_GC_par.loc[i + 1, "par_C"] * molecule_new[i] for i in range(220)]) + df_GC_par.loc[221, "par_C"]
        D_pre = np.sum([df_GC_par.loc[i + 1, "par_D"] * molecule_new[i] for i in range(220)]) + df_GC_par.loc[221, "par_D"]
        return A_pre,B_pre,C_pre,D_pre,list(sorted_JSC.values())[0]
    print(n)
    # n=5


    molecule_training=[df_candidatetraining_cpg.loc[list(sorted_JSC.keys())[i], [f"Group {j + 1}" for j in range(220)]].values for i in range(n)]
    property_training_Ag=[df_candidatetraining_cpg.loc[list(sorted_JSC.keys())[i], "Ag"] for i in range(n)]
    property_training_Bg=[df_candidatetraining_cpg.loc[list(sorted_JSC.keys())[i], "Bg"] for i in range(n)]
    property_training_Cg=[df_candidatetraining_cpg.loc[list(sorted_JSC.keys())[i], "Cg"] for i in range(n)]
    property_training_Dg=[df_candidatetraining_cpg.loc[list(sorted_JSC.keys())[i], "Dg"] for i in range(n)]
    property_training_Eg=[df_candidatetraining_cpg.loc[list(sorted_JSC.keys())[i], "Eg"] for i in range(n)]
    property_training_TMin=[df_candidatetraining_cpg.loc[list(sorted_JSC.keys())[i], "TMin"] for i in range(n)]
    property_training_TMax=[df_candidatetraining_cpg.loc[list(sorted_JSC.keys())[i], "TMax"] for i in range(n)]

    model = ConcreteModel()
    model.I = Set(initialize=[i for i in range(1, 5)])
    model.J = Set(initialize=[i for i in range(1, 221)])
    model.par = Var(model.I, model.J, within=Reals, initialize=0)
    model.par0 = Var(model.I, within=Reals, initialize=0)
    def obj_rule(model):
        obj = 0
        for i in range(n):
            A = model.par0[1]
            B = model.par0[2]
            C = model.par0[3]
            D = model.par0[4]
            for j in range(220):
                A += model.par[1, j+1] * molecule_training[i][j]
                B += model.par[2, j+1] * molecule_training[i][j]
                C += model.par[3, j+1] * molecule_training[i][j]
                D += model.par[4, j+1] * molecule_training[i][j]
            for t in range(11):
                T=property_training_TMin[i]+(property_training_TMax[i]-property_training_TMin[i])/10*t
                obj += ((A+B*(T/100)+C*(T/100)**2+D*(T/100)**3)-(property_training_Ag[i]+property_training_Bg[i]*(property_training_Cg[i]/T/np.sinh(property_training_Cg[i]/T))**2+property_training_Dg[i]*(property_training_Eg[i]/T/np.cosh(property_training_Eg[i]/T))**2)/1000)**2

        return obj
    model.obj = Objective(rule=obj_rule, sense=minimize)
    opt = SolverFactory('gams')
    io_options = dict()
    io_options['solver'] = "baron"
    io_options['mtype'] = "NLP"
    io_options['add_options'] = {'option resLim = 100; option optcr= 1e-6;'}
    io_options['warmstart'] = True
    result = opt.solve(model, tee=False, keepfiles=False, io_options=io_options)
    A_pre=np.sum([value(model.par[1,i+1])*molecule_new[i] for i in range(220)])+value(model.par0[1])
    B_pre=np.sum([value(model.par[2,i+1])*molecule_new[i] for i in range(220)])+value(model.par0[2])
    C_pre=np.sum([value(model.par[3,i+1])*molecule_new[i] for i in range(220)])+value(model.par0[3])
    D_pre=np.sum([value(model.par[4,i+1])*molecule_new[i] for i in range(220)])+value(model.par0[4])
    return A_pre,B_pre,C_pre,D_pre,list(sorted_JSC.values())[0]



for i in range(1,len(df_targetmol.index)+1):
    print(i)
    molecule_new = df_targetmol.loc[i, [f"Group {j + 1}" for j in range(220)]].values
    Ag_pre,Bg_pre,Cg_pre,Dg_pre,JSC_new=Linear(molecule_new, molecule)
    print(Ag_pre, Bg_pre, Cg_pre, Dg_pre,JSC_new)
    df_step3_out.loc[i, "Ag"] = Ag_pre
    df_step3_out.loc[i, "Bg"] = Bg_pre/100
    df_step3_out.loc[i, "Cg"] = Cg_pre/10000
    df_step3_out.loc[i, "Dg"] = Dg_pre/1000000
    df_step3_out.loc[i, "JSC_cpg"] = JSC_new


for i in df_step3_out.index:
    df_step3_out.loc[i,"pc_pre"]=df_step3_out.loc[i,"pc_pre"]*100000








#.......................................screen molecule(step4)..........................................................

for i in df_step3_out.index:
    df_step3_out.loc[i, "w_pre"] =(-np.log(df_step3_out.loc[i, "pc_pre"] * 0.98692327 * 1e-5) - 5.92714 + 6.09648 / (df_step3_out.loc[i, "tb_pre"] / df_step3_out.loc[i, "tc_pre"]) + 1.28862 * np.log(
    (df_step3_out.loc[i, "tb_pre"] / df_step3_out.loc[i, "tc_pre"])) - 0.169347 * (df_step3_out.loc[i, "tb_pre"] / df_step3_out.loc[i, "tc_pre"]) ** 6) / (15.2518 - 15.6875 / (df_step3_out.loc[i, "tb_pre"] / df_step3_out.loc[i, "tc_pre"]) - 13.4721 * np.log(
    (df_step3_out.loc[i, "tb_pre"] / df_step3_out.loc[i, "tc_pre"])) + 0.43577 * (df_step3_out.loc[i, "tb_pre"] / df_step3_out.loc[i, "tc_pre"]) ** 6);


for i in df_step3_out.index:
    if (df_step3_out.loc[i,"tb_pre"]>313 or df_step3_out.loc[i,"tc_pre"]<333.15):
        df_step3_out=df_step3_out.drop(index=i)
df1=df_step3_out["Ag"]
df2=df_step3_out["Bg"]
df3=df_step3_out["Cg"]
df4=df_step3_out["Dg"]
df5=df_step3_out["tc_pre"]
df6=df_step3_out["tb_pre"]
df7=df_step3_out["pc_pre"]
df8=df_step3_out["w_pre"]
df9=pd.DataFrame(range(1,len(df_step3_out)+1),index=df_step3_out.index,columns=["name"])

new_excel=pd.ExcelWriter(fr"{filepath_input_process}")
df1.to_excel(new_excel,sheet_name="An")
df2.to_excel(new_excel,sheet_name="Bn")
df3.to_excel(new_excel,sheet_name="Cn")
df4.to_excel(new_excel,sheet_name="Dn")
df5.to_excel(new_excel,sheet_name="Tcn")
df6.to_excel(new_excel,sheet_name="Tbn")
df7.to_excel(new_excel,sheet_name="Pcn")
df8.to_excel(new_excel,sheet_name="wn")
df9.to_excel(new_excel,sheet_name="name")

new_excel.close()





















