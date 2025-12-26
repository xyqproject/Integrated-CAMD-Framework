$oneolcom
$eolcom //

$title Solvent Design

$ontext
MINLP model for CAMD problem using Marrero & Gani's group contribution method.
$offtext

// first order groups
$gdxin GC_Coef_MG_GP1.gdx
sets
    i0(*)   "groups for Marrero & Gani GC method"
    i1(i0)  "groups selected in the molecule"
;

$load i0

// candidate groups
//i1(i0) = no;
//i1('CH3') = yes;
//i1('CH2') = yes;
//i1('CH') = yes;
//i1('C') = yes;
//i1('OH') = yes;
//i1('CHO') = yes;
//i1('CH3COO') = yes;
//i1('CH2COO') = yes;
//i1('HCOO') = yes;
//i1('CH3CO') = yes;
//i1('CH2CO') = yes;
//i1('COOH') = yes;
//i1('COO') = yes;





i1('CH2') = yes;
i1('CH3') = yes;
i1('CH') = yes;
i1('OH') = yes;
i1('CH2=CH') = yes;
i1('C') = yes;
i1('COOH') = yes;
i1('CH2=C') = yes;
i1('CH3COO') = yes;
i1('CH2O') = yes;
i1('CH2NH2') = yes;
i1('CH3CO') = yes;
i1('CH2Cl') = yes;
i1('CH=CH') = yes;
i1('CH3O') = yes;
i1('CH2CN') = yes;
i1('CH2SH') = yes;
i1('CH2COO') = yes;
i1('CHO') = yes;
i1('COO') = yes;


alias(i1,ii1);
// parameters

parameters
    Noc(i0)     "group type (valency)"
    Mw(i0)      "molecular weight [g/mol]"

;

$load Noc Mw

$gdxin

scalar R0               "gas constant [J/mol/K]"                /8.314/;
scalar T0               "temperature [K]"                       /298.15/;
scalar EP               "eps"                                   /1e-6/;


scalar TotGrp_U         "upper bound of total No. groups"       /8/;
scalar TotGrp_L         "lower bound of total No. groups"       /2/;
scalar FuncGrp_U        "upper bound of func. groups"           /2/;
scalar FuncGrp_L        "lower bound of func. groups"           /0/;
scalar SameGrp_U        "upper bound of same groups"            /2/;

scalar Mw_U     "upper bound of molecular weight [g/Mol]"       /150/;
scalar Mw_L     "lower bound of molecular weight [g/Mol]"       /100/;
*------------------------------------------------------------------------------*

// variables

integer variables
        n1(i0)  "number of 1st order groups"
        q       "q=-1->bicyclic; q=0->monocyclic; q=1->acyclic"
;





// upper and lower bounds
n1.lo(i0) = 0;
n1.up(i0) = SameGrp_U;

q.lo = 1;
q.up = 1;

variables
        dummy           "objective value"
;

*------------------------------------------------------------------------------*

// equations

equations
        edummy      "dummy objective function"

        val_0       "valency constraints"
        val_1(i0)   "valency constraints"

        // constraints for group numbers
        n_g_l       "minimum number of groups"
        n_g_u       "maximum number of groups"
        n_f_l       "minimum number of functional groups"
        n_f_u       "maximum number of functional groups"

        // constraints for physical properties
        mw_up       "upper bound of Mw"
        mw_low       "lower bound of Mw"
;

edummy..        dummy =e= 1;

val_0..         sum(i1, (2-Noc(i1))*n1(i1)) =e= 2*q;
val_1(i1)..     sum(ii1$(not sameas(i1, ii1)), n1(ii1))
                    =g= (Noc(i1)-2)*n1(i1) + 2;

// constraints for group numbers
n_g_l..         sum(i1, n1(i1)) =g= TotGrp_L;
n_g_u..         sum(i1, n1(i1)) =l= TotGrp_U;
n_f_l..         sum(i1, n1(i1))
                    - n1('CH3') - n1('CH2') - n1('CH') - n1('C')
                    =g= FuncGrp_L;
n_f_u..         sum(i1, n1(i1))
                    - n1('CH3') - n1('CH2') - n1('CH') - n1('C')
                    =l= FuncGrp_U;

// constraints for physical properties
mw_up..         sum(i1, n1(i1)*Mw(i1)) =l= Mw_U;
mw_low..         sum(i1, n1(i1)*Mw(i1)) =g= Mw_L;

*------------------------------------------------------------------------------*

// solver options

model WorkingFluid /edummy,val_0,val_1,n_g_l,n_g_u,n_f_l,n_f_u,
                    mw_up,mw_low/;





option optcr = 0;
option reslim = 1e6;

// baron options
$onecho > baron.opt
numsol 5000
gdxout mult
$offecho

WorkingFluid.optfile = 1;

$offlisting
$offsymxref offsymlist

option
    limrow = 0,
    limcol = 0,
    solprint = off,
    sysout = off
;

set sol "possible feasible solutions"   /mult1*mult100000/
    solnpool(sol)   "all feasible solutions"
;

variables nx(sol,i0), qx(sol);

option mip = baron;

solve WorkingFluid using mip minimizing dummy;




execute 'gdxmerge mult*.gdx > %gams.scrdir%merge.%gams.scrext%';
execute "del /f mult*";

execute_load 'merged.gdx', nx = n1, qx = q, solnpool = Merged_set_1;

scalar cardsoln  "number of feasible solutions";

cardsoln = card(solnpool);
    //display cardsoln;

display nx.l;
display qx.l,solnpool,i1;
Execute_Unload 'step1_output_case2.gdx',nx;
Execute 'Gdxxrw step1_output_case2.gdx O = step1_output_case2.xlsx var = nx Rng =sheet1!a1:zz100000';































































