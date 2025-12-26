$oneolcom
$eolcom //

$title Working Fluids Design



sets
         compound1  /List_refrigerant1/
         compound2  /List_refrigerant2/
         An_ /Ag/
         Bn_ /Bg/
         Cn_ /Cg/
         Dn_ /Dg/
         wn_ /w_pre/
         Tbn_ /tb_pre/
         Tcn_ /tc_pre/
         Pcn_ /pc_pre/
         name /name/
         ;
Parameters
    list_An1(compound1,An_), list_Bn1(compound1,Bn_), list_Cn1(compound1,Cn_), list_Dn1(compound1,Dn_), list_wn1(compound1,wn_), list_Tbn1(compound1,Tbn_),list_Tcn1(compound1,Tcn_), list_Pcn1(compound1,Pcn_), list_name1(compound1,name),
    list_An2(compound2,An_), list_Bn2(compound2,Bn_), list_Cn2(compound2,Cn_), list_Dn2(compound2,Dn_), list_wn2(compound2,wn_), list_Tbn2(compound2,Tbn_),list_Tcn2(compound2,Tcn_), list_Pcn2(compound2,Pcn_), list_name2(compound2,name);

$onecho > input.txt
par=list_An1 rng=An!A1:B425 Rdim=1 Cdim=1
par=list_Bn1 rng=Bn!A1:B425 Rdim=1 Cdim=1
par=list_Cn1 rng=Cn!A1:B425 Rdim=1 Cdim=1
par=list_Dn1 rng=Dn!A1:B425 Rdim=1 Cdim=1
par=list_wn1 rng=wn!A1:B425 Rdim=1 Cdim=1
par=list_Tcn1 rng=Tcn!A1:B425 Rdim=1 Cdim=1
par=list_Tbn1 rng=Tbn!A1:B425 Rdim=1 Cdim=1
par=list_Pcn1 rng=Pcn!A1:B425 Rdim=1 Cdim=1
par=list_name1 rng=name!A1:B425 Rdim=1 Cdim=1
par=list_An2 rng=An!A1:B425 Rdim=1 Cdim=1
par=list_Bn2 rng=Bn!A1:B425 Rdim=1 Cdim=1
par=list_Cn2 rng=Cn!A1:B425 Rdim=1 Cdim=1
par=list_Dn2 rng=Dn!A1:B425 Rdim=1 Cdim=1
par=list_wn2 rng=wn!A1:B425 Rdim=1 Cdim=1
par=list_Tcn2 rng=Tcn!A1:B425 Rdim=1 Cdim=1
par=list_Tbn2 rng=Tbn!A1:B425 Rdim=1 Cdim=1
par=list_Pcn2 rng=Pcn!A1:B425 Rdim=1 Cdim=1
par=list_name2 rng=name!A1:B425 Rdim=1 Cdim=1
$offecho

$call gdxxrw filepath_input_process.xlsx output=filepath_input_process.gdx @input.txt
$gdxin filepath_input_process.gdx

$load  list_An1 list_Bn1 list_Cn1 list_Dn1 list_wn1 list_Tcn1 list_Tbn1 list_Pcn1 list_name1 list_An2 list_Bn2 list_Cn2 list_Dn2 list_wn2 list_Tcn2 list_Tbn2 list_Pcn2 list_name2
$gdxin
display  list_An1,list_Bn1,list_Cn1,list_Dn1,list_wn1,list_Tcn1,list_Tbn1,list_Pcn1,list_name1,list_An2,list_Bn2,list_Cn2,list_Dn2,list_wn2,list_Tcn2,list_Tbn2,list_Pcn2,list_name2;




parameters T0,R0,P0, H0l, S0l, H0v, S0v, EoS,EoSsigma, EoSepsilon, EoSomega, EoSpsi, EoSkappa1, EoSkappa2,Te;
parameters An1, Bn1, Cn1, Dn1, wn1, Tbn1, Tcn1, Pcn1, Mn1;
parameters An2, Bn2, Cn2, Dn2, wn2, Tbn2, Tcn2, Pcn2, Mn2;
parameter table_all(compound1,compound2);


// Reference
R0=8.314;
P0 = 101325;  //lee and kessler
T0 = 273.15;
H0l = 0;//200*Mn;
S0l = 0;//1*Mn;
H0v = 0;//H0l + HvT0n*((1-T0/Tcn)/(1-298.15/Tcn))**0.38;
S0v = 0;//S0l + HvT0n*((1-T0/Tcn)/(1-298.15/Tcn))**0.38/T0;
EoS = 1; // Select EoS, 1=SRK, 2=PR
Te=260.15;


positive variables t1, t2, t3, t4,t5,t6,t7,t8,p1, p2, p3, p4,p5,p6,p7,p8,zl1, zv1, zv2, zl3, zv3,zv4,zl5,zv5,zl7,zv7;
variables cop;

equations
    objcop
    eqzv1     "Vap Z1 calc"
    eqzv2     "Vap Z2 calc"
    eqzv3
    eqzv4
    eqzv5
    eqzv7
    eqzl1
    eqzl3
    eqzl5
    eqzl7     "Liq Z7 calc"
    eqh1     "residual enthalpy h1"
    eqh2     "residual enthalpy h2"
    eqh3     "residual enthalpy h3"
    eqs12
    eqs34
    eqphi1
    eqphi3
    eqphi5
    eqphi7
    eqcond1
    eqcond2
    eqcond3
    eqcond4
    eqcond5
    eqcond6
    eqcond7
    eqcond8
    eqcond9
    eqcond10
    eqcond11
    eqcond12
    eqcond13
    eqcond14
    eqcond15
    eqcond16
    eqcond17
    eqcond18
    eqcond19
    eqcond20
    eqcond21
    eqcond22
    eqcond23
    eqcond24
    eqcond25
    eqcond26
    eqcond27
    eqcond28
    eqcond29
    eqcond30
    eqcond31
    eqcond32
    eqcond33
    eqcond34
    eqcond35
    eqcond36
    eqcond37
    eqcond38
    eqcond39
    eqcond40
    eqcond41
    eqcond42
    eqcond43
    eqcond44
    eqcond45
;


objcop.. cop =e= (( H0v + (1/4)*Dn2*(t3**4-T0**4)+(1/3)*Cn2*(t3**3-T0**3)+(1/2)*Bn2*(t3**2-T0**2)+An2*(t3-T0) + R0*t3*(zv3-1)+(-t3*(1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2*EoSkappa2/(Pcn2*sqrt((1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*t3*Tcn2))-(1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2/Pcn2)*Pcn2*log((zv3+EoSomega*Tcn2*p3/(Pcn2*t3))/zv3)/(EoSomega*R0*Tcn2))
                - ( H0l + (1/4)*Dn2*(t5**4-T0**4)+(1/3)*Cn2*(t5**3-T0**3)+(1/2)*Bn2*(t5**2-T0**2)+An2*(t5-T0) + R0*t5*(zl5-1)+(-t5*(1+EoSkappa2*(1-sqrt(t5/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2*EoSkappa2/(Pcn2*sqrt((1+EoSkappa2*(1-sqrt(t5/Tcn2)))**2*t5*Tcn2))-(1+EoSkappa2*(1-sqrt(t5/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2/Pcn2)*Pcn2*log((zl5+EoSomega*Tcn2*p5/(Pcn2*t5))/zl5)/(EoSomega*R0*Tcn2)))*
                 (( H0v + (1/4)*Dn1*(t1**4-T0**4)+(1/3)*Cn1*(t1**3-T0**3)+(1/2)*Bn1*(t1**2-T0**2)+An1*(t1-T0) + R0*t1*(zv1-1)+(-t1*(1+EoSkappa1*(1-sqrt(t1/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2*EoSkappa1/(Pcn1*sqrt((1+EoSkappa1*(1-sqrt(t1/Tcn1)))**2*t1*Tcn1))-(1+EoSkappa1*(1-sqrt(t1/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2/Pcn1)*Pcn1*log((zv1+EoSomega*Tcn1*p1/(Pcn1*t1))/zv1)/(EoSomega*R0*Tcn1))
                - ( H0l + (1/4)*Dn1*(t7**4-T0**4)+(1/3)*Cn1*(t7**3-T0**3)+(1/2)*Bn1*(t7**2-T0**2)+An1*(t7-T0) + R0*t7*(zl7-1)+(-t7*(1+EoSkappa1*(1-sqrt(t7/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2*EoSkappa1/(Pcn1*sqrt((1+EoSkappa1*(1-sqrt(t7/Tcn1)))**2*t7*Tcn1))-(1+EoSkappa1*(1-sqrt(t7/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2/Pcn1)*Pcn1*log((zl7+EoSomega*Tcn1*p7/(Pcn1*t7))/zl7)/(EoSomega*R0*Tcn1)))
               /(
                 (( H0v + (1/4)*Dn2*(t3**4-T0**4)+(1/3)*Cn2*(t3**3-T0**3)+(1/2)*Bn2*(t3**2-T0**2)+An2*(t3-T0) + R0*t3*(zv3-1)+(-t3*(1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2*EoSkappa2/(Pcn2*sqrt((1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*t3*Tcn2))-(1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2/Pcn2)*Pcn2*log((zv3+EoSomega*Tcn2*p3/(Pcn2*t3))/zv3)/(EoSomega*R0*Tcn2))
                - ( H0l + (1/4)*Dn2*(t5**4-T0**4)+(1/3)*Cn2*(t5**3-T0**3)+(1/2)*Bn2*(t5**2-T0**2)+An2*(t5-T0) + R0*t5*(zl5-1)+(-t5*(1+EoSkappa2*(1-sqrt(t5/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2*EoSkappa2/(Pcn2*sqrt((1+EoSkappa2*(1-sqrt(t5/Tcn2)))**2*t5*Tcn2))-(1+EoSkappa2*(1-sqrt(t5/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2/Pcn2)*Pcn2*log((zl5+EoSomega*Tcn2*p5/(Pcn2*t5))/zl5)/(EoSomega*R0*Tcn2)))*
                 (( H0v + (1/4)*Dn1*(t2**4-T0**4)+(1/3)*Cn1*(t2**3-T0**3)+(1/2)*Bn1*(t2**2-T0**2)+An1*(t2-T0) + R0*t2*(zv2-1)+(-t2*(1+EoSkappa1*(1-sqrt(t2/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2*EoSkappa1/(Pcn1*sqrt((1+EoSkappa1*(1-sqrt(t2/Tcn1)))**2*t2*Tcn1))-(1+EoSkappa1*(1-sqrt(t2/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2/Pcn1)*Pcn1*log((zv2+EoSomega*Tcn1*p2/(Pcn1*t2))/zv2)/(EoSomega*R0*Tcn1))
                - ( H0v + (1/4)*Dn1*(t1**4-T0**4)+(1/3)*Cn1*(t1**3-T0**3)+(1/2)*Bn1*(t1**2-T0**2)+An1*(t1-T0) + R0*t1*(zv1-1)+(-t1*(1+EoSkappa1*(1-sqrt(t1/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2*EoSkappa1/(Pcn1*sqrt((1+EoSkappa1*(1-sqrt(t1/Tcn1)))**2*t1*Tcn1))-(1+EoSkappa1*(1-sqrt(t1/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2/Pcn1)*Pcn1*log((zv1+EoSomega*Tcn1*p1/(Pcn1*t1))/zv1)/(EoSomega*R0*Tcn1)))
                +(( H0v + (1/4)*Dn1*(t2**4-T0**4)+(1/3)*Cn1*(t2**3-T0**3)+(1/2)*Bn1*(t2**2-T0**2)+An1*(t2-T0) + R0*t2*(zv2-1)+(-t2*(1+EoSkappa1*(1-sqrt(t2/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2*EoSkappa1/(Pcn1*sqrt((1+EoSkappa1*(1-sqrt(t2/Tcn1)))**2*t2*Tcn1))-(1+EoSkappa1*(1-sqrt(t2/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2/Pcn1)*Pcn1*log((zv2+EoSomega*Tcn1*p2/(Pcn1*t2))/zv2)/(EoSomega*R0*Tcn1))
                - ( H0l + (1/4)*Dn1*(t7**4-T0**4)+(1/3)*Cn1*(t7**3-T0**3)+(1/2)*Bn1*(t7**2-T0**2)+An1*(t7-T0) + R0*t7*(zl7-1)+(-t7*(1+EoSkappa1*(1-sqrt(t7/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2*EoSkappa1/(Pcn1*sqrt((1+EoSkappa1*(1-sqrt(t7/Tcn1)))**2*t7*Tcn1))-(1+EoSkappa1*(1-sqrt(t7/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2/Pcn1)*Pcn1*log((zl7+EoSomega*Tcn1*p7/(Pcn1*t7))/zl7)/(EoSomega*R0*Tcn1)))*
                 (( H0v + (1/4)*Dn2*(t4**4-T0**4)+(1/3)*Cn2*(t4**3-T0**3)+(1/2)*Bn2*(t4**2-T0**2)+An2*(t4-T0) + R0*t4*(zv4-1)+(-t4*(1+EoSkappa2*(1-sqrt(t4/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2*EoSkappa2/(Pcn2*sqrt((1+EoSkappa2*(1-sqrt(t4/Tcn2)))**2*t4*Tcn2))-(1+EoSkappa2*(1-sqrt(t4/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2/Pcn2)*Pcn2*log((zv4+EoSomega*Tcn2*p4/(Pcn2*t4))/zv4)/(EoSomega*R0*Tcn2))
                - ( H0v + (1/4)*Dn2*(t3**4-T0**4)+(1/3)*Cn2*(t3**3-T0**3)+(1/2)*Bn2*(t3**2-T0**2)+An2*(t3-T0) + R0*t3*(zv3-1)+(-t3*(1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2*EoSkappa2/(Pcn2*sqrt((1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*t3*Tcn2))-(1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2/Pcn2)*Pcn2*log((zv3+EoSomega*Tcn2*p3/(Pcn2*t3))/zv3)/(EoSomega*R0*Tcn2)))
                );

// Compressability
eqzl1..  0 =e= zl1**3-zl1**2+zl1*(-(EoSomega*Tcn1*p1/(Pcn1*t1))**2+EoSpsi*(1+EoSkappa1*(1-sqrt(t1/Tcn1)))**2*Tcn1*Tcn1*p1/(Pcn1*t1*t1)-EoSomega*Tcn1*p1/(Pcn1*t1))-EoSpsi*(1+EoSkappa1*(1-sqrt(t1/Tcn1)))**2*Tcn1*Tcn1*Tcn1*p1*p1*EoSomega/(Pcn1*Pcn1*t1*t1*t1);
eqzl3..  0 =e= zl3**3-zl3**2+zl3*(-(EoSomega*Tcn2*p3/(Pcn2*t3))**2+EoSpsi*(1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*Tcn2*Tcn2*p3/(Pcn2*t3*t3)-EoSomega*Tcn2*p3/(Pcn2*t3))-EoSpsi*(1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*Tcn2*Tcn2*Tcn2*p3*p3*EoSomega/(Pcn2*Pcn2*t3*t3*t3);
eqzl5..  0 =e= zl5**3-zl5**2+zl5*(-(EoSomega*Tcn2*p5/(Pcn2*t5))**2+EoSpsi*(1+EoSkappa2*(1-sqrt(t5/Tcn2)))**2*Tcn2*Tcn2*p5/(Pcn2*t5*t5)-EoSomega*Tcn2*p5/(Pcn2*t5))-EoSpsi*(1+EoSkappa2*(1-sqrt(t5/Tcn2)))**2*Tcn2*Tcn2*Tcn2*p5*p5*EoSomega/(Pcn2*Pcn2*t5*t5*t5);
eqzl7..  0 =e= zl7**3-zl7**2+zl7*(-(EoSomega*Tcn1*p7/(Pcn1*t7))**2+EoSpsi*(1+EoSkappa1*(1-sqrt(t7/Tcn1)))**2*Tcn1*Tcn1*p7/(Pcn1*t7*t7)-EoSomega*Tcn1*p7/(Pcn1*t7))-EoSpsi*(1+EoSkappa1*(1-sqrt(t7/Tcn1)))**2*Tcn1*Tcn1*Tcn1*p7*p7*EoSomega/(Pcn1*Pcn1*t7*t7*t7);

eqzv1..  0 =e= zv1**3-zv1**2+zv1*(-(EoSomega*Tcn1*p1/(Pcn1*t1))**2+EoSpsi*(1+EoSkappa1*(1-sqrt(t1/Tcn1)))**2*Tcn1*Tcn1*p1/(Pcn1*t1*t1)-EoSomega*Tcn1*p1/(Pcn1*t1))-EoSpsi*(1+EoSkappa1*(1-sqrt(t1/Tcn1)))**2*Tcn1*Tcn1*Tcn1*p1*p1*EoSomega/(Pcn1*Pcn1*t1*t1*t1);
eqzv2..  0 =e= zv2**3-zv2**2+zv2*(-(EoSomega*Tcn1*p2/(Pcn1*t2))**2+EoSpsi*(1+EoSkappa1*(1-sqrt(t2/Tcn1)))**2*Tcn1*Tcn1*p2/(Pcn1*t2*t2)-EoSomega*Tcn1*p2/(Pcn1*t2))-EoSpsi*(1+EoSkappa1*(1-sqrt(t2/Tcn1)))**2*Tcn1*Tcn1*Tcn1*p2*p2*EoSomega/(Pcn1*Pcn1*t2*t2*t2);
eqzv3..  0 =e= zv3**3-zv3**2+zv3*(-(EoSomega*Tcn2*p3/(Pcn2*t3))**2+EoSpsi*(1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*Tcn2*Tcn2*p3/(Pcn2*t3*t3)-EoSomega*Tcn2*p3/(Pcn2*t3))-EoSpsi*(1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*Tcn2*Tcn2*Tcn2*p3*p3*EoSomega/(Pcn2*Pcn2*t3*t3*t3);
eqzv4..  0 =e= zv4**3-zv4**2+zv4*(-(EoSomega*Tcn2*p4/(Pcn2*t4))**2+EoSpsi*(1+EoSkappa2*(1-sqrt(t4/Tcn2)))**2*Tcn2*Tcn2*p4/(Pcn2*t4*t4)-EoSomega*Tcn2*p4/(Pcn2*t4))-EoSpsi*(1+EoSkappa2*(1-sqrt(t4/Tcn2)))**2*Tcn2*Tcn2*Tcn2*p4*p4*EoSomega/(Pcn2*Pcn2*t4*t4*t4);
eqzv5..  0 =e= zv5**3-zv5**2+zv5*(-(EoSomega*Tcn2*p5/(Pcn2*t5))**2+EoSpsi*(1+EoSkappa2*(1-sqrt(t5/Tcn2)))**2*Tcn2*Tcn2*p5/(Pcn2*t5*t5)-EoSomega*Tcn2*p5/(Pcn2*t5))-EoSpsi*(1+EoSkappa2*(1-sqrt(t5/Tcn2)))**2*Tcn2*Tcn2*Tcn2*p5*p5*EoSomega/(Pcn2*Pcn2*t5*t5*t5);
eqzv7..  0 =e= zv7**3-zv7**2+zv7*(-(EoSomega*Tcn1*p7/(Pcn1*t7))**2+EoSpsi*(1+EoSkappa1*(1-sqrt(t7/Tcn1)))**2*Tcn1*Tcn1*p7/(Pcn1*t7*t7)-EoSomega*Tcn1*p7/(Pcn1*t7))-EoSpsi*(1+EoSkappa1*(1-sqrt(t7/Tcn1)))**2*Tcn1*Tcn1*Tcn1*p7*p7*EoSomega/(Pcn1*Pcn1*t7*t7*t7);

eqs12..  S0v + (1/3)*Dn1*t1*t1*t1+(1/2)*Cn1*t1*t1+Bn1*t1+An1*log(t1)-(1/3)*Dn1*T0*T0*T0-(1/2)*Cn1*T0*T0-Bn1*T0-An1*log(T0) - R0*log(p1/P0) + R0*log(zv1-EoSomega*Tcn1*p1/(Pcn1*t1))-R0*Tcn1*(1+EoSkappa1*(1-sqrt(t1/Tcn1)))**2*EoSpsi*EoSkappa1*log((zv1+EoSomega*Tcn1*p1/(Pcn1*t1))/zv1)/(EoSomega*sqrt((1+EoSkappa1*(1-sqrt(t1/Tcn1)))**2*t1*Tcn1))
              =e=
         S0v + (1/3)*Dn1*t2*t2*t2+(1/2)*Cn1*t2*t2+Bn1*t2+An1*log(t2)-(1/3)*Dn1*T0*T0*T0-(1/2)*Cn1*T0*T0-Bn1*T0-An1*log(T0) - R0*log(p2/P0) + R0*log(zv2-EoSomega*Tcn1*p2/(Pcn1*t2))-R0*Tcn1*(1+EoSkappa1*(1-sqrt(t2/Tcn1)))**2*EoSpsi*EoSkappa1*log((zv2+EoSomega*Tcn1*p2/(Pcn1*t2))/zv2)/(EoSomega*sqrt((1+EoSkappa1*(1-sqrt(t2/Tcn1)))**2*t2*Tcn1));
eqs34..  S0v + (1/3)*Dn2*t3*t3*t3+(1/2)*Cn2*t3*t3+Bn2*t3+An2*log(t3)-(1/3)*Dn2*T0*T0*T0-(1/2)*Cn2*T0*T0-Bn2*T0-An2*log(T0) - R0*log(p3/P0) + R0*log(zv3-EoSomega*Tcn2*p3/(Pcn2*t3))-R0*Tcn2*(1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*EoSpsi*EoSkappa2*log((zv3+EoSomega*Tcn2*p3/(Pcn2*t3))/zv3)/(EoSomega*sqrt((1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*t3*Tcn2))
              =e=
         S0v + (1/3)*Dn2*t4*t4*t4+(1/2)*Cn2*t4*t4+Bn2*t4+An2*log(t4)-(1/3)*Dn2*T0*T0*T0-(1/2)*Cn2*T0*T0-Bn2*T0-An2*log(T0) - R0*log(p4/P0) + R0*log(zv4-EoSomega*Tcn2*p4/(Pcn2*t4))-R0*Tcn2*(1+EoSkappa2*(1-sqrt(t4/Tcn2)))**2*EoSpsi*EoSkappa2*log((zv4+EoSomega*Tcn2*p4/(Pcn2*t4))/zv4)/(EoSomega*sqrt((1+EoSkappa2*(1-sqrt(t4/Tcn2)))**2*t4*Tcn2));

// fugacity
eqphi1.. exp(zv1-1-log(zv1-EoSomega*Tcn1*p1/(Pcn1*t1))-EoSpsi*(1+EoSkappa1*(1-sqrt(t1/Tcn1)))**2*Tcn1*log((zv1+EoSomega*Tcn1*p1/(Pcn1*t1))/zv1)/(t1*EoSomega))
     =e= exp(zl1-1-log(zl1-EoSomega*Tcn1*p1/(Pcn1*t1))-EoSpsi*(1+EoSkappa1*(1-sqrt(t1/Tcn1)))**2*Tcn1*log((zl1+EoSomega*Tcn1*p1/(Pcn1*t1))/zl1)/(t1*EoSomega));
eqphi3.. exp(zv3-1-log(zv3-EoSomega*Tcn2*p3/(Pcn2*t3))-EoSpsi*(1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*Tcn2*log((zv3+EoSomega*Tcn2*p3/(Pcn2*t3))/zv3)/(t3*EoSomega))
     =e= exp(zl3-1-log(zl3-EoSomega*Tcn2*p3/(Pcn2*t3))-EoSpsi*(1+EoSkappa2*(1-sqrt(t3/Tcn2)))**2*Tcn2*log((zl3+EoSomega*Tcn2*p3/(Pcn2*t3))/zl3)/(t3*EoSomega));
eqphi5.. exp(zv5-1-log(zv5-EoSomega*Tcn2*p5/(Pcn2*t5))-EoSpsi*(1+EoSkappa2*(1-sqrt(t5/Tcn2)))**2*Tcn2*log((zv5+EoSomega*Tcn2*p5/(Pcn2*t5))/zv5)/(t5*EoSomega))
     =e= exp(zl5-1-log(zl5-EoSomega*Tcn2*p5/(Pcn2*t5))-EoSpsi*(1+EoSkappa2*(1-sqrt(t5/Tcn2)))**2*Tcn2*log((zl5+EoSomega*Tcn2*p5/(Pcn2*t5))/zl5)/(t5*EoSomega));
eqphi7.. exp(zv7-1-log(zv7-EoSomega*Tcn1*p7/(Pcn1*t7))-EoSpsi*(1+EoSkappa1*(1-sqrt(t7/Tcn1)))**2*Tcn1*log((zv7+EoSomega*Tcn1*p7/(Pcn1*t7))/zv7)/(t7*EoSomega))
     =e= exp(zl7-1-log(zl7-EoSomega*Tcn1*p7/(Pcn1*t7))-EoSpsi*(1+EoSkappa1*(1-sqrt(t7/Tcn1)))**2*Tcn1*log((zl7+EoSomega*Tcn1*p7/(Pcn1*t7))/zl7)/(t7*EoSomega));


// operating conditions
eqcond1..   p1 =e= p8;
eqcond2..   p2 =e= p7;
eqcond3..   p3 =e= p6;
eqcond4..   p4 =e= p5;
eqcond5..   t6 =e= t3;
eqcond6..   t8 =e= t1;
eqcond7..   t1 =l= t2;
eqcond8..   t3 =l= t4;
eqcond9..   t7 =l= t5;
eqcond10..   t5 =l= t4;
eqcond11..   t2 =l= Tcn1;       //Tcn
eqcond12..   t4 =l= Tcn2;
eqcond13..   p1 =g= 100000;     //90000
eqcond14..   p1 =l= p2;
eqcond15..   p3 =g= 100000;     //90000  *
eqcond16..   p3 =l= p4;
eqcond17..   p2 =l= Pcn1;       //Pcn
eqcond18..   p4 =l= Pcn2;       //Pcn
eqcond19..  t1 =g= Te-15;       //270
eqcond20..  t3 =g= Tbn2;       //270
eqcond21..  0  =l= 3*zv1*zv1-2*zv1-rPower(EoSomega*Tcn1*p1/(Pcn1*t1),2)+EoSpsi*rPower(1+EoSkappa1*(1-sqrt(t1/Tcn1)), 2)*Tcn1*Tcn1*p1/(Pcn1*t1*t1)-EoSomega*Tcn1*p1/(Pcn1*t1);
eqcond22..  0  =l= 3*zl1*zl1-2*zl1-rPower(EoSomega*Tcn1*p1/(Pcn1*t1),2)+EoSpsi*rPower(1+EoSkappa1*(1-sqrt(t1/Tcn1)), 2)*Tcn1*Tcn1*p1/(Pcn1*t1*t1)-EoSomega*Tcn1*p1/(Pcn1*t1);
eqcond23..  0  =l= 3*zv3*zv3-2*zv3-rPower(EoSomega*Tcn2*p3/(Pcn2*t3),2)+EoSpsi*rPower(1+EoSkappa2*(1-sqrt(t3/Tcn2)), 2)*Tcn2*Tcn2*p3/(Pcn2*t3*t3)-EoSomega*Tcn2*p3/(Pcn2*t3);
eqcond24..  0  =l= 3*zl3*zl3-2*zl3-rPower(EoSomega*Tcn2*p3/(Pcn2*t3),2)+EoSpsi*rPower(1+EoSkappa2*(1-sqrt(t3/Tcn2)), 2)*Tcn2*Tcn2*p3/(Pcn2*t3*t3)-EoSomega*Tcn2*p3/(Pcn2*t3);
eqcond25..  0  =l= 3*zv5*zv5-2*zv5-rPower(EoSomega*Tcn2*p5/(Pcn2*t5),2)+EoSpsi*rPower(1+EoSkappa2*(1-sqrt(t5/Tcn2)), 2)*Tcn2*Tcn2*p5/(Pcn2*t5*t5)-EoSomega*Tcn2*p5/(Pcn2*t5);
eqcond26..  0  =l= 3*zl5*zl5-2*zl5-rPower(EoSomega*Tcn2*p5/(Pcn2*t5),2)+EoSpsi*rPower(1+EoSkappa2*(1-sqrt(t5/Tcn2)), 2)*Tcn2*Tcn2*p5/(Pcn2*t5*t5)-EoSomega*Tcn2*p5/(Pcn2*t5);
eqcond27..  0  =l= 3*zv7*zv7-2*zv7-rPower(EoSomega*Tcn1*p7/(Pcn1*t7),2)+EoSpsi*rPower(1+EoSkappa1*(1-sqrt(t7/Tcn1)), 2)*Tcn1*Tcn1*p7/(Pcn1*t7*t7)-EoSomega*Tcn1*p7/(Pcn1*t7);
eqcond28..  0  =l= 3*zl7*zl7-2*zl7-rPower(EoSomega*Tcn1*p7/(Pcn1*t7),2)+EoSpsi*rPower(1+EoSkappa1*(1-sqrt(t7/Tcn1)), 2)*Tcn1*Tcn1*p7/(Pcn1*t7*t7)-EoSomega*Tcn1*p7/(Pcn1*t7);
eqcond29..  0  =l= 6*zv1-2;
eqcond30..  0  =g= 6*zl1-2;
eqcond31..  0  =l= 6*zv3-2;
eqcond32..  0  =g= 6*zl3-2;
eqcond33..  0  =l= 6*zv5-2;
eqcond34..  0  =g= 6*zl5-2;
eqcond35..  0  =l= 6*zv7-2;
eqcond36..  0  =g= 6*zl7-2;
eqcond37..  0  =l= 3*zv2*zv2-2*zv2-rPower(EoSomega*Tcn1*p2/(Pcn1*t2),2)+EoSpsi*rPower(1+EoSkappa1*(1-sqrt(t2/Tcn1)), 2)*Tcn1*Tcn1*p2/(Pcn1*t2*t2)-EoSomega*Tcn1*p2/(Pcn1*t2);
eqcond38..  0  =l= 6*zv2-2;
eqcond39..  0  =l= 3*zv4*zv4-2*zv4-rPower(EoSomega*Tcn2*p4/(Pcn2*t4),2)+EoSpsi*rPower(1+EoSkappa2*(1-sqrt(t4/Tcn2)), 2)*Tcn2*Tcn2*p4/(Pcn2*t4*t4)-EoSomega*Tcn2*p4/(Pcn2*t4);
eqcond40..  0  =l= 6*zv4-2;
eqcond41..  t7 =g= t1;
eqcond42..  t7 =g= t3;
eqcond43..  t7 =l= t2;
eqcond44..  t5 =g= t3;       //270
eqcond45..  t3 =g= t1;

model ztp /objcop,
          eqzl1,eqzl3, eqzl5,eqzl7,
          eqzv1, eqzv2, eqzv3, eqzv4,eqzv5,eqzv7,
          eqs12, eqs34,
          eqphi1, eqphi3,  eqphi5,eqphi7,
          eqcond1, eqcond2, eqcond3,
          eqcond4, eqcond5,
          eqcond6, eqcond7,
          eqcond8,
          eqcond9,
          eqcond10,
          eqcond11, eqcond12, eqcond13, eqcond14, eqcond15, eqcond16, eqcond17, eqcond18, eqcond19, eqcond20,
          eqcond21, eqcond22, eqcond23, eqcond24, eqcond25, eqcond26, eqcond27, eqcond28, eqcond29, eqcond30,
          eqcond31, eqcond32, eqcond33, eqcond34, eqcond35, eqcond36, eqcond37, eqcond38, eqcond39, eqcond40, eqcond41, eqcond42, eqcond43, eqcond44, eqcond45
          /;


$onecho > knitro.opt
maxit 100000
xtol 1e-1000
honorBnds 1
ms_enable 0
//feastol 1e-3
$offecho
option nlp = knitro;
ztp.optfile = 1;





loop(compound1,
                 An1=list_An1(compound1,"Ag");
                 Bn1=list_Bn1(compound1,"Bg");
                 Cn1=list_Cn1(compound1,"Cg");
                 Dn1=list_Dn1(compound1,"Dg");
                 wn1=list_wn1(compound1,"w_pre");
                 Tcn1=list_Tcn1(compound1,"tc_pre");
                 Tbn1=list_Tbn1(compound1,"tb_pre");
                 Pcn1=list_Pcn1(compound1,"pc_pre");

                 loop(compound2,
                                   An2=list_An2(compound2,"Ag");
                                   Bn2=list_Bn2(compound2,"Bg");
                                   Cn2=list_Cn2(compound2,"Cg");
                                   Dn2=list_Dn2(compound2,"Dg");
                                   wn2=list_wn2(compound2,"w_pre");
                                   Tcn2=list_Tcn2(compound2,"tc_pre");
                                   Tbn2=list_Tbn2(compound2,"tb_pre");
                                   Pcn2=list_Pcn2(compound2,"pc_pre");

                                   if((Tbn1<Te-15)and(Tbn2<Tcn1)and(333.15<Tcn2)and(Tbn2<313.15),

                                                                    // Select EoS
                                                                    if(EoS = 1,
                                                                       EoSsigma   = 1;
                                                                       EoSepsilon = 0;
                                                                       EoSomega   = 0.08664;
                                                                       EoSpsi     = 0.42748;
                                                                       EoSkappa1   = 0.480+1.574*wn1-0.176*wn1*wn1;
                                                                       EoSkappa2   = 0.480+1.574*wn2-0.176*wn2*wn2;
                                                                    elseif(EoS = 2),
                                                                       EoSsigma   = 1 + sqrt(2);
                                                                       EoSepsilon = 1 - sqrt(2);
                                                                       EoSomega   = 0.0778;
                                                                       EoSpsi     = 0.45724;
                                                                       EoSkappa1   = 0.37464+1.54226*wn1-0.26992*wn1*wn1;
                                                                       EoSkappa2   = 0.37464+1.54226*wn2-0.26992*wn2*wn2;
                                                                    );



                                                                    //  Initial guess
                                                                    t1.l = Te-10;
                                                                    t2.l = Tbn2+10;
                                                                    t3.l = Tbn2;
                                                                    t4.l = 330;
                                                                    t5.l = 320;
                                                                    t6.l = Tbn2;
                                                                    t7.l = Tbn2;
                                                                    t8.l = Te-10;
                                                                    p1.l = P0;
                                                                    p2.l = 2*P0;
                                                                    p3.l = 3*P0;
                                                                    p4.l = 5*P0;
                                                                    p5.l = 5*P0;
                                                                    p6.l = 3*P0;
                                                                    p7.l = 2*P0;
                                                                    p8.l = P0;
                                                                    zv1.l = 1;
                                                                    zv2.l = 1;
                                                                    zv3.l = 1;
                                                                    zv4.l = 1;
                                                                    zv5.l = 1;
                                                                    zv7.l = 1;
                                                                    zl1.l = EoSomega*Tcn1*p1.l/(Pcn1*t1.l) + 0.01;
                                                                    zl3.l = EoSomega*Tcn2*p3.l/(Pcn2*t3.l) + 0.01;
                                                                    zl5.l = EoSomega*Tcn2*p5.l/(Pcn2*t5.l) + 0.01;
                                                                    zl7.l = EoSomega*Tcn1*p7.l/(Pcn1*t7.l) + 0.01;
                                                                    cop.l=5;

                                                                    //bound
                                                                    t1.lo=Te-15;t1.up=Te;
                                                                    t2.lo=Tbn2;t2.up=500;
                                                                    t3.lo=Tbn2;t3.up=500;
                                                                    t4.lo=313.15;t4.up=500;
                                                                    t5.lo=313.15;t5.up=333.15;
                                                                    t6.lo=Tbn2;t6.up=500;
                                                                    t7.lo=Tbn2;t7.up=500;
                                                                    t8.lo=Te-15;t8.up=Te;
                                                                    p1.lo=100000;P1.up=1000000;
                                                                    p2.lo=100000;P2.up=3000000;
                                                                    p3.lo=100000;P3.up=3000000;
                                                                    p4.lo=100000;P4.up=3000000;
                                                                    p5.lo=100000;P5.up=3000000;
                                                                    p6.lo=100000;P6.up=3000000;
                                                                    p7.lo=100000;P7.up=3000000;
                                                                    p8.lo=100000;P8.up=1000000;
                                                                    zv1.lo = 1/3;zv1.up=1;
                                                                    zv2.lo = 1/3;zv2.up=1;
                                                                    zv3.lo = 1/3;zv3.up=1;
                                                                    zv4.lo = 1/3;zv4.up=1;
                                                                    zv5.lo = 1/3;zv5.up=1;
                                                                    zv7.lo = 1/3;zv7.up=1;
                                                                    zl1.lo = 0;zl1.up=1/3;
                                                                    zl3.lo = 0;zl3.up=1/3;
                                                                    zl5.lo = 0;zl5.up=1/3;
                                                                    zl7.lo = 0;zl7.up=1/3;
                                                                    cop.lo=0;cop.up=10;


                                                                    solve ztp using nlp maximising cop;
                                                                    if((zv1.l>0.34)and(zv2.l>0.34)and(zv3.l>0.34)and(zv4.l>0.34)and(zv5.l>0.34)and(zv7.l>0.34)and(zl1.l<0.32)and(zl3.l<0.32)and(zl5.l<0.32)and(zl7.l<0.32),
                                                                                                         table_all(compound1,compound2)=cop.l;); );
                         );
         );




display cop.l,t1.l,t2.l,t7.l,t8.l,t3.l,t4.l,t5.l,t6.l,p1.l,p2.l,p7.l,p8.l,p3.l,p4.l,p5.l,p6.l,zl1.l,zl3.l,zl5.l,zl7.l,zv1.l,zv3.l,zv5.l,zv7.l,zv2.l,zv4.l,table_all;



parameter h1,h2,h3,h4,h5,h7,optcop,phil1,phiv1,phil3,phiv3,phil5,phiv5,phil7,phiv7,s1,s2,s3,s4;
h1=( H0v + (1/4)*Dn1*(t1.l**4-T0**4)+(1/3)*Cn1*(t1.l**3-T0**3)+(1/2)*Bn1*(t1.l**2-T0**2)+An1*(t1.l-T0) + R0*t1.l*(zv1.l-1)+(-t1.l*(1+EoSkappa1*(1-sqrt(t1.l/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2*EoSkappa1/(Pcn1*sqrt((1+EoSkappa1*(1-sqrt(t1.l/Tcn1)))**2*t1.l*Tcn1))-(1+EoSkappa1*(1-sqrt(t1.l/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2/Pcn1)*Pcn1*log((zv1.l+EoSomega*Tcn1*p1.l/(Pcn1*t1.l))/zv1.l)/(EoSomega*R0*Tcn1));
h2=( H0v + (1/4)*Dn1*(t2.l**4-T0**4)+(1/3)*Cn1*(t2.l**3-T0**3)+(1/2)*Bn1*(t2.l**2-T0**2)+An1*(t2.l-T0) + R0*t2.l*(zv2.l-1)+(-t2.l*(1+EoSkappa1*(1-sqrt(t2.l/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2*EoSkappa1/(Pcn1*sqrt((1+EoSkappa1*(1-sqrt(t2.l/Tcn1)))**2*t2.l*Tcn1))-(1+EoSkappa1*(1-sqrt(t2.l/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2/Pcn1)*Pcn1*log((zv2.l+EoSomega*Tcn1*p2.l/(Pcn1*t2.l))/zv2.l)/(EoSomega*R0*Tcn1));
h3=( H0v + (1/4)*Dn2*(t3.l**4-T0**4)+(1/3)*Cn2*(t3.l**3-T0**3)+(1/2)*Bn2*(t3.l**2-T0**2)+An2*(t3.l-T0) + R0*t3.l*(zv3.l-1)+(-t3.l*(1+EoSkappa2*(1-sqrt(t3.l/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2*EoSkappa2/(Pcn2*sqrt((1+EoSkappa2*(1-sqrt(t3.l/Tcn2)))**2*t3.l*Tcn2))-(1+EoSkappa2*(1-sqrt(t3.l/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2/Pcn2)*Pcn2*log((zv3.l+EoSomega*Tcn2*p3.l/(Pcn2*t3.l))/zv3.l)/(EoSomega*R0*Tcn2));
h4=( H0v + (1/4)*Dn2*(t4.l**4-T0**4)+(1/3)*Cn2*(t4.l**3-T0**3)+(1/2)*Bn2*(t4.l**2-T0**2)+An2*(t4.l-T0) + R0*t4.l*(zv4.l-1)+(-t4.l*(1+EoSkappa2*(1-sqrt(t4.l/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2*EoSkappa2/(Pcn2*sqrt((1+EoSkappa2*(1-sqrt(t4.l/Tcn2)))**2*t4.l*Tcn2))-(1+EoSkappa2*(1-sqrt(t4.l/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2/Pcn2)*Pcn2*log((zv4.l+EoSomega*Tcn2*p4.l/(Pcn2*t4.l))/zv4.l)/(EoSomega*R0*Tcn2));
h5=( H0l + (1/4)*Dn2*(t5.l**4-T0**4)+(1/3)*Cn2*(t5.l**3-T0**3)+(1/2)*Bn2*(t5.l**2-T0**2)+An2*(t5.l-T0) + R0*t5.l*(zl5.l-1)+(-t5.l*(1+EoSkappa2*(1-sqrt(t5.l/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2*EoSkappa2/(Pcn2*sqrt((1+EoSkappa2*(1-sqrt(t5.l/Tcn2)))**2*t5.l*Tcn2))-(1+EoSkappa2*(1-sqrt(t5.l/Tcn2)))**2*EoSpsi*R0**2*Tcn2**2/Pcn2)*Pcn2*log((zl5.l+EoSomega*Tcn2*p5.l/(Pcn2*t5.l))/zl5.l)/(EoSomega*R0*Tcn2));
h7=( H0l + (1/4)*Dn1*(t7.l**4-T0**4)+(1/3)*Cn1*(t7.l**3-T0**3)+(1/2)*Bn1*(t7.l**2-T0**2)+An1*(t7.l-T0) + R0*t7.l*(zl7.l-1)+(-t7.l*(1+EoSkappa1*(1-sqrt(t7.l/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2*EoSkappa1/(Pcn1*sqrt((1+EoSkappa1*(1-sqrt(t7.l/Tcn1)))**2*t7.l*Tcn1))-(1+EoSkappa1*(1-sqrt(t7.l/Tcn1)))**2*EoSpsi*R0**2*Tcn1**2/Pcn1)*Pcn1*log((zl7.l+EoSomega*Tcn1*p7.l/(Pcn1*t7.l))/zl7.l)/(EoSomega*R0*Tcn1));

optcop=(h3-h5)*(h1-h7)/((h3-h5)*(h2-h1)+(h2-h7)*(h4-h3));

phil1=exp(zl1.l-1-log(zl1.l-EoSomega*Tcn1*p1.l/(Pcn1*t1.l))-EoSpsi*(1+EoSkappa1*(1-sqrt(t1.l/Tcn1)))**2*Tcn1*log((zl1.l+EoSomega*Tcn1*p1.l/(Pcn1*t1.l))/zl1.l)/(t1.l*EoSomega));
phiv1=exp(zv1.l-1-log(zv1.l-EoSomega*Tcn1*p1.l/(Pcn1*t1.l))-EoSpsi*(1+EoSkappa1*(1-sqrt(t1.l/Tcn1)))**2*Tcn1*log((zv1.l+EoSomega*Tcn1*p1.l/(Pcn1*t1.l))/zv1.l)/(t1.l*EoSomega));
phil3=exp(zl3.l-1-log(zl3.l-EoSomega*Tcn2*p3.l/(Pcn2*t3.l))-EoSpsi*(1+EoSkappa2*(1-sqrt(t3.l/Tcn2)))**2*Tcn2*log((zl3.l+EoSomega*Tcn2*p3.l/(Pcn2*t3.l))/zl3.l)/(t3.l*EoSomega));
phiv3=exp(zv3.l-1-log(zv3.l-EoSomega*Tcn2*p3.l/(Pcn2*t3.l))-EoSpsi*(1+EoSkappa2*(1-sqrt(t3.l/Tcn2)))**2*Tcn2*log((zv3.l+EoSomega*Tcn2*p3.l/(Pcn2*t3.l))/zv3.l)/(t3.l*EoSomega));
phil5=exp(zl5.l-1-log(zl5.l-EoSomega*Tcn2*p5.l/(Pcn2*t5.l))-EoSpsi*(1+EoSkappa2*(1-sqrt(t5.l/Tcn2)))**2*Tcn2*log((zl5.l+EoSomega*Tcn2*p5.l/(Pcn2*t5.l))/zl5.l)/(t5.l*EoSomega));
phiv5=exp(zv5.l-1-log(zv5.l-EoSomega*Tcn2*p5.l/(Pcn2*t5.l))-EoSpsi*(1+EoSkappa2*(1-sqrt(t5.l/Tcn2)))**2*Tcn2*log((zv5.l+EoSomega*Tcn2*p5.l/(Pcn2*t5.l))/zv5.l)/(t5.l*EoSomega));
phil7=exp(zl7.l-1-log(zl7.l-EoSomega*Tcn1*p7.l/(Pcn1*t7.l))-EoSpsi*(1+EoSkappa1*(1-sqrt(t7.l/Tcn1)))**2*Tcn1*log((zl7.l+EoSomega*Tcn1*p7.l/(Pcn1*t7.l))/zl7.l)/(t7.l*EoSomega));
phiv7=exp(zv7.l-1-log(zv7.l-EoSomega*Tcn1*p7.l/(Pcn1*t7.l))-EoSpsi*(1+EoSkappa1*(1-sqrt(t7.l/Tcn1)))**2*Tcn1*log((zv7.l+EoSomega*Tcn1*p7.l/(Pcn1*t7.l))/zv7.l)/(t7.l*EoSomega));



s1=S0v + (1/3)*Dn1*t1.l*t1.l*t1.l+(1/2)*Cn1*t1.l*t1.l+Bn1*t1.l+An1*log(t1.l)-(1/3)*Dn1*T0*T0*T0-(1/2)*Cn1*T0*T0-Bn1*T0-An1*log(T0) - R0*log(p1.l/P0) + R0*log(zv1.l-EoSomega*Tcn1*p1.l/(Pcn1*t1.l))-R0*Tcn1*(1+EoSkappa1*(1-sqrt(t1.l/Tcn1)))**2*EoSpsi*EoSkappa1*log((zv1.l+EoSomega*Tcn1*p1.l/(Pcn1*t1.l))/zv1.l)/(EoSomega*sqrt((1+EoSkappa1*(1-sqrt(t1.l/Tcn1)))**2*t1.l*Tcn1));
s2=S0v + (1/3)*Dn1*t2.l*t2.l*t2.l+(1/2)*Cn1*t2.l*t2.l+Bn1*t2.l+An1*log(t2.l)-(1/3)*Dn1*T0*T0*T0-(1/2)*Cn1*T0*T0-Bn1*T0-An1*log(T0) - R0*log(p2.l/P0) + R0*log(zv2.l-EoSomega*Tcn1*p2.l/(Pcn1*t2.l))-R0*Tcn1*(1+EoSkappa1*(1-sqrt(t2.l/Tcn1)))**2*EoSpsi*EoSkappa1*log((zv2.l+EoSomega*Tcn1*p2.l/(Pcn1*t2.l))/zv2.l)/(EoSomega*sqrt((1+EoSkappa1*(1-sqrt(t2.l/Tcn1)))**2*t2.l*Tcn1));
s3=S0v + (1/3)*Dn2*t3.l*t3.l*t3.l+(1/2)*Cn2*t3.l*t3.l+Bn2*t3.l+An2*log(t3.l)-(1/3)*Dn2*T0*T0*T0-(1/2)*Cn2*T0*T0-Bn2*T0-An2*log(T0) - R0*log(p3.l/P0) + R0*log(zv3.l-EoSomega*Tcn2*p3.l/(Pcn2*t3.l))-R0*Tcn2*(1+EoSkappa2*(1-sqrt(t3.l/Tcn2)))**2*EoSpsi*EoSkappa2*log((zv3.l+EoSomega*Tcn2*p3.l/(Pcn2*t3.l))/zv3.l)/(EoSomega*sqrt((1+EoSkappa2*(1-sqrt(t3.l/Tcn2)))**2*t3.l*Tcn2));
s4=S0v + (1/3)*Dn2*t4.l*t4.l*t4.l+(1/2)*Cn2*t4.l*t4.l+Bn2*t4.l+An2*log(t4.l)-(1/3)*Dn2*T0*T0*T0-(1/2)*Cn2*T0*T0-Bn2*T0-An2*log(T0) - R0*log(p4.l/P0) + R0*log(zv4.l-EoSomega*Tcn2*p4.l/(Pcn2*t4.l))-R0*Tcn2*(1+EoSkappa2*(1-sqrt(t4.l/Tcn2)))**2*EoSpsi*EoSkappa2*log((zv4.l+EoSomega*Tcn2*p4.l/(Pcn2*t4.l))/zv4.l)/(EoSomega*sqrt((1+EoSkappa2*(1-sqrt(t4.l/Tcn2)))**2*t4.l*Tcn2));






display h1,h2,h3,h4,h5,h7,optcop,phil1,phiv1,phil3,phiv3,phil5,phiv5,phil7,phiv7,s1,s2,s3,s4;
Execute_Unload 'file_result.gdx',table_all;
Execute 'Gdxxrw file_result.gdx O = file_result.xlsx par = table_all Rng =B!a1:zz2000';







