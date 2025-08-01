#pragma once
#ifndef PERIODIC_TABLE_HPP
#define PERIODIC_TABLE_HPP

namespace ptable{

//*********************************************
//Periodic Table Data
//*********************************************

static const int N_ELEMENTS=118;

static const char* const NAME[N_ELEMENTS]={
	"H",
	"He",
	"Li",
	"Be",
	"B",
	"C",
	"N",
	"O",
	"F",
	"Ne",
	"Na",
	"Mg",
	"Al",
	"Si",
	"P",
	"S",
	"Cl",
	"Ar",
	"K",
	"Ca",
	"Sc",
	"Ti",
	"V",
	"Cr",
	"Mn",
	"Fe",
	"Ni",
	"Co",
	"Cu",
	"Zn",
	"Ga",
	"Ge",
	"As",
	"Se",
	"Br",
	"Xe",
	"Rb",
	"Sr",
	"Y",
	"Zr",
	"Nb",
	"Mo",
	"Tc",
	"Ru",
	"Rh",
	"Pd",
	"Ag",
	"Cd",
	"In",
	"Sn",
	"Sb",
	"Te",
	"I",
	"Xe",
	"Cs",
	"Ba",
	"La",
	"Ce",
	"Pr",
	"Nd",
	"Pm",
	"Sm",
	"Eu",
	"Gd",
	"Tb",
	"Dy",
	"Ho",
	"Er",
	"Tm",
	"Yb",
	"Lu",
	"Hf",
	"Ta",
	"W",
	"Re",
	"Os",
	"Ir",
	"Pt",
	"Au",
	"Hg",
	"Tl",
	"Pb",
	"Bi",
	"Po",
	"At",
	"Rn",
	"Fr",
	"Ra",
	"Ac",
	"Th",
	"Pa",
	"U",
	"Np",
	"Pu",
	"Am",
	"Cm",
	"Bk",
	"Cf",
	"Es",
	"Fm",
	"Md",
	"No",
	"Lr",
	"Rf",
	"Db",
	"Sg",
	"Bh",
	"Hs",
	"Mt",
	"Ds",
	"Rg",
	"Cn",
	"Nh",
	"Fl",
	"Mc",
	"Lv",
	"Ts",
	"Og"
};

static const double MASS[N_ELEMENTS]={
	1.0079, //H
	4.0026, //He
	6.941, //Li
	9.0122, //Be
	10.811, //B
	12.0107, //C
	14.0067, //N
	15.9994, //O
	18.9984, //F
	20.1797, //Ne
	22.9897, //Na
	24.305, //Mg
	26.9815, //Al
	28.0855, //Si
	30.9738, //P
	32.065, //S
	35.453, //Cl
	39.0983, //Ar
	39.948, //K
	40.078, //Ca
	44.9559, //Sc
	47.867, //Ti
	50.9415, //V
	51.9961, //Cr
	54.938, //Mn
	55.845, //Fe
	58.6934, //Co
	58.9332, //Ni
	63.546, //Cu
	65.39, //Zn
	69.723, //Ga
	72.64, //Ge
	74.9216, //As
	78.96, //Se
	79.904, //Br
	83.8, //Kr
	85.4678, //Rb
	87.62, //Sr
	88.9059, //Y
	91.224, //Zr
	92.9064, //Nb
	95.94, //Mo
	98, //Tc
	101.07, //Ru
	102.9055, //Rh
	106.42, //Pd
	107.8682, //Ag
	112.411, //Cd
	114.818, //In
	118.71, //Sn
	121.76, //Sb
	126.9045, //Te
	127.6, //I
	131.293, //Xe
	132.9055, //Cs
	137.327, //Ba
	138.9055, //La
	140.116, //Ce
	140.9077, //Pr
	144.24, //Nd
	145, //Pm
	150.36, //Sm
	151.964, //Eu
	157.25, //Gd
	158.9253, //Tb
	162.5, //Dy
	164.9303, //Ho
	167.259, //Er
	168.9342, //Tm
	173.04, //Yb
	174.967, //Lu
	178.49, //Hf
	180.9479, //Ta
	183.84, //W
	186.207, //Re
	190.23, //Os
	192.217, //Ir
	195.078, //Pt
	196.9665, //Au
	200.59, //Hg
	204.3833, //Tl
	207.2, //Pb
	208.9804, //Bi
	209, //Po
	210, //At
	222, //Rn
	223, //Fr
	226, //Ra
	227, //Ac
	231.0359, //Th
	232.0381, //Pa
	237, //U
	238.0289, //Np
	243, //Pu
	244, //Am
	247, //Cm
	247, //Bk
	251, //Cf
	252, //Es
	257, //Fm
	258, //Md
	259, //No
	261, //Lr
	262, //Rf
	262, //Db
	264, //Sg
	266, //Bh
	268, //Hs
	272, //Mt
	277, //Ds
	278, //Rg
	281, //Cn
	282, //Nh
	285, //Fl
	286, //Mc
	289, //Lv
	290, //Ts
	293 //Og
};

static const double RADIUS_COVALENT[N_ELEMENTS]={
	0.31, //H
	0.28, //He
	1.28, //Li
	0.96, //Be
	0.84, //B
	0.76, //C
	0.71, //N
	0.66, //O
	0.57, //F
	0.58, //Ne
	1.66, //Na
	1.41, //Mg
	1.21, //Al
	1.11, //Si
	1.07, //P
	1.05, //S
	1.02, //Cl
	1.06, //Ar
	2.03, //K
	1.76, //Ca
	1.7, //Sc
	1.6, //Ti
	1.53, //V
	1.39, //Cr
	1.39, //Mn
	1.32, //Fe
	1.26, //Co
	1.24, //Ni
	1.32, //Cu
	1.22, //Zn
	1.22, //Ga
	1.2, //Ge
	1.19, //As
	1.2, //Se
	1.2, //Br
	1.16, //Kr
	2.2, //Rb
	1.95, //Sr
	1.9, //Y
	1.75, //Zr
	1.64, //Nb
	1.54, //Mo
	1.47, //Tc
	1.46, //Ru
	1.42, //Rh
	1.39, //Pd
	1.45, //Ag
	1.44, //Cd
	1.42, //In
	1.39, //Sn
	1.39, //Sb
	1.38, //Te
	1.39, //I
	1.4, //Xe
	2.44, //Cs
	2.15, //Ba
	2.07, //La
	2.04, //Ce
	2.03, //Pr
	2.01, //Nd
	1.99, //Pm
	1.98, //Sm
	1.98, //Eu
	1.96, //Gd
	1.94, //Tb
	1.92, //Dy
	1.92, //Ho
	1.89, //Er
	1.9, //Tm
	1.87, //Yb
	1.87, //Lu
	1.75, //Hf
	1.7, //Ta
	1.62, //W
	1.51, //Re
	1.44, //Os
	1.41, //Ir
	1.36, //Pt
	1.36, //Au
	1.32, //Hg
	1.45, //Tl
	1.46, //Pb
	1.48, //Bi
	1.4, //Po
	1.5, //At
	1.5, //Rn
	2.6, //Fr
	2.21, //Ra
	2.15, //Ac
	2.06, //Th
	2, //Pa
	1.96, //U
	1.9, //Np
	1.87, //Pu
	1.8, //Am
	1.69, //Cm
	-1, //Bk
	-1, //Cf
	-1, //Es
	-1, //Fm
	-1, //Md
	-1, //No
	-1, //Lr
	-1, //Rf
	-1, //Db
	-1, //Sg
	-1, //Bh
	-1, //Hs
	-1, //Mt
	-1, //Ds
	-1, //Rg
	-1, //Cn
	-1, //Nh
	-1, //Fl
	-1, //Mc
	-1, //Lv
	-1, //Ts
	-1, //Og
};
//B. Cordero, V. Gómez, A.E. Platero-Prats, M. Revés, J. Echeverría, E. Cremades, F. Barragán, and S. Alvarez, Dalton Trans. 2832 (2008)

static const double RADIUS_COVALENT_IE[N_ELEMENTS]={
	0.42, //H
	0.31, //He
	1.28, //Li
	0.97, //Be
	0.98, //B
	0.84, //C
	0.73, //N
	0.72, //O
	0.64, //F
	0.58, //Ne
	1.55, //Na
	1.34, //Mg
	1.48, //Al
	1.30, //Si
	1.16, //P
	1.13, //S
	1.03, //Cl
	0.95, //Ar
	1.76, //K
	1.72, //Ca
	1.63, //Sc
	1.42, //Ti
	1.35, //V
	1.43, //Cr
	1.31, //Mn
	1.26, //Fe
	1.27, //Co
	1.10, //Ni
	1.58, //Cu
	1.41, //Zn
	1.55, //Ga
	1.43, //Ge
	1.30, //As
	1.32, //Se
	1.27, //Br
	1.14, //Kr
	2.17, //Rb
	1.90, //Sr
	1.73, //Y
	1.69, //Zr
	1.57, //Nb
	1.64, //Mo
	1.53, //Tc
	1.44, //Ru
	1.38, //Rh
	1.28, //Pd
	1.75, //Ag
	1.62, //Cd
	1.82, //In
	1.62, //Sn
	1.43, //Sb
	1.44, //Te
	1.32, //I
	1.10, //Xe
	0.00, //Cs
	0.00, //Ba
	0.00, //La
	0.00, //Ce
	0.00, //Pr
	0.00, //Nd
	0.00, //Pm
	0.00, //Sm
	0.00, //Eu
	0.00, //Gd
	0.00, //Tb
	0.00, //Dy
	0.00, //Ho
	0.00, //Er
	0.00, //Tm
	0.00, //Yb
	0.00, //Lu
	0.00, //Hf
	0.00, //Ta
	0.00, //W
	0.00, //Re
	0.00, //Os
	0.00, //Ir
	0.00, //Pt
	0.00, //Au
	0.00, //Hg
	0.00, //Tl
	0.00, //Pb
	0.00, //Bi
	0.00, //Po
	0.00, //At
	0.00, //Rn
	0.00, //Fr
	0.00, //Ra
	0.00, //Ac
	0.00, //Th
	0.00, //Pa
	0.00, //U
	0.00, //Np
	0.00, //Pu
	0.00, //Am
	0.00, //Cm
	0.00, //Bk
	0.00, //Cf
	0.00, //Es
	0.00, //Fm
	0.00, //Md
	0.00, //No
	0.00, //Lr
	0.00, //Rf
	0.00, //Db
	0.00, //Sg
	0.00, //Bh
	0.00, //Hs
	0.00, //Mt
	0.00, //Ds
	0.00, //Rg
	0.00, //Cn
	0.00, //Nh
	0.00, //Fl
	0.00, //Mc
	0.00, //Lv
	0.00, //Ts
	0.00  //Og
};
//B. Cordero, V. Gómez, A.E. Platero-Prats, M. Revés, J. Echeverría, E. Cremades, F. Barragán, and S. Alvarez, Dalton Trans. 2832 (2008)

static const double RADIUS_VDW[N_ELEMENTS]={
	1.10,//H
	1.40,//He
	1.81,//Li
	1.53,//Be
	1.92,//B
	1.70,//C
	1.55,//N
	1.52,//O
	1.47,//F
	1.54,//Ne
	2.27,//Na
	1.73,//Mg
	1.84,//Al
	2.10,//Si
	1.80,//P
	1.80,//S
	1.75,//Cl
	1.88,//Ar
	2.75,//K
	2.31,//Ca
	0.0,//Sc
	0.0,//Ti
	0.0,//V
	0.0,//Cr
	0.0,//Mn
	0.0,//Fe
	0.0,//Co
	0.0,//Ni
	0.0,//Cu
	0.0,//Zn
	1.87,//Ga
	2.11,//Ge
	1.85,//As
	1.90,//Se
	1.83,//Br
	2.02,//Kr
	3.03,//Rb
	2.49,//Sr
	0.0,//Y
	0.0,//Zr
	0.0,//Nb
	0.0,//Mo
	0.0,//Tc
	0.0,//Ru
	0.0,//Rh
	0.0,//Pd
	0.0,//Ag
	0.0,//Cd
	1.93,//In
	2.17,//Sn
	2.06,//Sb
	2.06,//Te
	1.98,//I
	2.16,//Xe
	3.43,//Cs
	2.68,//Ba
	0.0,//La
	0.0,//Ce
	0.0,//Pr
	0.0,//Nd
	0.0,//Pm
	0.0,//Sm
	0.0,//Eu
	0.0,//Gd
	0.0,//Tb
	0.0,//Dy
	0.0,//Ho
	0.0,//Er
	0.0,//Tm
	0.0,//Yb
	0.0,//Lu
	0.0,//Hf
	0.0,//Ta
	0.0,//W
	0.0,//Re
	0.0,//Os
	0.0,//Ir
	0.0,//Pt
	0.0,//Au
	0.0,//Hg
	1.96,//Tl
	2.02,//Pb
	2.07,//Bi
	1.97,//Po
	2.02,//At
	2.20,//Rn
	3.48,//Fr
	2.83,//Ra
	0.0,//Ac
	0.0,//Th
	0.0,//Pa
	0.0,//U
	0.0,//Np
	0.0,//Pu
	0.0,//Am
	0.0,//Cm
	0.0,//Bk
	0.0,//Cf
	0.0,//Es
	0.0,//Fm
	0.0,//Md
	0.0,//No
	0.0,//Lr
	0.0,//Rf
	0.0,//Db
	0.0,//Sg
	0.0,//Bh
	0.0,//Hs
	0.0,//Mt
	0.0,//Ds
	0.0,//Rg
	0.0,//Cn
	0.0,//Nh
	0.0,//Fl
	0.0,//Mc
	0.0,//Lv
	0.0,//Ts
	0.0//Og
};
//Consistent van der Waals Radii for the Whole Main Group

static const double IONIZATION_ENERGY[N_ELEMENTS]={
	13.598434005136, //H
	24.587387936, //He
	5.391714761, //Li
	9.322699, //Be
	8.298019, //B
	11.2603, //C
	14.53413, //N
	13.618054, //O
	17.42282, //F
	21.56454, //Ne
	5.1390767, //Na
	7.646235, //Mg
	5.985768, //Al
	8.151683, //Si
	10.486686, //P
	10.36001, //S
	12.96763, //Cl
	15.7596112, //Ar
	4.34066354, //K
	6.1131552, //Ca
	6.56149, //Sc
	6.82812, //Ti
	6.746187, //V
	6.76651, //Cr
	7.434038, //Mn
	7.9024678, //Fe
	7.88101, //Ni
	7.639877, //Co
	7.72638, //Cu
	9.3941968, //Zn
	5.9993018, //Ga
	7.899435, //Ge
	9.7886, //As
	9.752392, //Se
	11.81381, //Br
	13.9996049, //Xe
	4.177128, //Rb
	5.6948672, //Sr
	6.21726, //Y
	6.6339, //Zr
	6.75885, //Nb
	7.09243, //Mo
	7.11938, //Tc
	7.3605, //Ru
	7.4589, //Rh
	8.33686, //Pd
	7.576234, //Ag
	8.99382, //Cd
	5.7863554, //In
	7.343917, //Sn
	8.608389, //Sb
	9.00966, //Te
	10.45126, //I
	12.1298431, //Xe
	3.893905557, //Cs
	5.211664, //Ba
	5.5769, //La
	5.5386, //Ce
	5.47, //Pr
	5.525, //Nd
	5.577, //Pm
	5.64371, //Sm
	5.670385, //Eu
	6.1498, //Gd
	5.8638, //Tb
	5.93905, //Dy
	6.0215, //Ho
	6.1077, //Er
	6.18431, //Tm
	6.254159, //Yb
	5.425871, //Lu
	6.825069, //Hf
	7.549571, //Ta
	7.86403, //W
	7.83352, //Re
	8.43823, //Os
	8.96702, //Ir
	8.95883, //Pt
	9.225553, //Au
	10.437504, //Hg
	6.1082871, //Tl
	7.4166796, //Pb
	7.285516, //Bi
	8.414, //Po
	9.31751, //At
	4.0727409, //Rn
	5.278424, //Fr
	5.380226, //Ra
	6.3067, //Ac
	5.89, //Th
	6.19405, //Pa
	6.2655, //U
	6.0258, //Np
	5.9738, //Pu
	5.9914, //Am
	6.1978, //Cm
	6.2817, //Bk
	6.3676, //Cf
	6.5, //Es
	6.58, //Fm
	6.65, //Md
	0, //No
	0, //Lr
	0, //Rf
	0, //Db
	0, //Sg
	0, //Bh
	0, //Hs
	0, //Mt
	0, //Ds
	0, //Rg
	0, //Cn
	0, //Nh
	0, //Fl
	0, //Mc
	0, //Lv
	0, //Ts
	0 //Og
};
//ionization energies are in eV
//Kramida, A., Ralchenko, Yu., Reader, J., and NIST ASD Team (2014). NIST Atomic Spectra Database (ver. 5.2), [Online]. 
//Available: https://physics.nist.gov/asd [2017, August 9]. National Institute of Standards and Technology, Gaithersburg, MD. 

static const double ELECTRON_AFFINITY[N_ELEMENTS]={
	0.754195, //H
	-0.52, //He
	0.618049, //Li
	-0.52, //Be
	0.279723, //B
	1.2621226, //C
	-0.000725, //N
	1.4611136, //O
	3.4011898, //F
	-1.2, //Ne
	0.547926, //Na
	-0.415, //Mg
	0.43283, //Al
	1.3895212, //Si
	0.746607, //P
	2.0771042, //S
	3.612724, //Cl
	-1, //Ar
	0.501459, //K
	0.02455, //Ca
	0.188, //Sc
	0.084, //Ti
	0.52766, //V
	0.67584, //Cr
	-0.52, //Mn
	0.153236, //Fe
	0.66226, //Ni
	1.15716, //Co
	1.23578, //Cu
	-0.62, //Zn
	0.43, //Ga
	1.2326764, //Ge
	0.8048, //As
	2.0206047, //Se
	3.363588, //Br
	-0.62, //Xe
	0.485916, //Rb
	0.05206, //Sr
	0.307, //Y
	0.427, //Zr
	0.9174, //Nb
	0.7473, //Mo
	0.55, //Tc
	1.04638, //Ru
	1.14289, //Rh
	0.56214, //Pd
	1.30447, //Ag
	-0.725, //Cd
	0.3, //In
	1.11207, //Sn
	1.047401, //Sb
	1.970875, //Te
	3.0590465, //I
	-0.83, //Xe
	0.47163, //Cs
	0.14462, //Ba
	0.47, //La
	0.65, //Ce
	0.962, //Pr
	1.916, //Nd
	0.129, //Pm
	0.162, //Sm
	0.864, //Eu
	0.137, //Gd
	1.165, //Tb
	0.352, //Dy
	0.338, //Ho
	0.312, //Er
	1.029, //Tm
	-0.02, //Yb
	0.346, //Lu
	0.017, //Hf
	0.323, //Ta
	0.81626, //W
	0.15, //Re
	1.1, //Os
	1.56436, //Ir
	2.1251, //Pt
	2.30861, //Au
	-0.52, //Hg
	0.377, //Tl
	0.356743, //Pb
	0.942362, //Bi
	1.9, //Po
	2.3, //At
	-0.725, //Rn
	0.486, //Fr
	0.1, //Ra
	0.35, //Ac
	1.17, //Th
	0.55, //Pa
	0.53, //U
	0.48, //Np
	-0.5, //Pu
	0.1, //Am
	0.28, //Cm
	-1.72, //Bk
	-1.01, //Cf
	-0.3, //Es
	0.35, //Fm
	0.98, //Md
	-2.33, //No
	-0.31, //Lr
	-0, //Rf
	-0, //Db
	-0, //Sg
	-0, //Bh
	-0, //Hs
	-0, //Mt
	-0, //Ds
	-0, //Rg
	-0, //Cn
	-0, //Nh
	-0, //Fl
	-0, //Mc
	-0, //Lv
	-0, //Ts
	-0 //Og
};
//electron affinities are in eV
//wikipedia

static const double ETA_UFF[N_ELEMENTS]={
	13.89, //H
	29.84, //He
	4.772, //Li
	8.886, //Be
	9.5, //B
	10.126, //C
	11.76, //N
	13.364, //O
	14.948, //F
	21.1, //Ne
	4.592, //Na
	7.386, //Mg
	7.18, //Al
	6.974, //Si
	8, //P
	8.972, //S
	9.892, //Cl
	12.71, //Ar
	3.84, //K
	5.76, //Ca
	6.16, //Sc
	6.76, //Ti
	6.82, //V
	7.73, //Cr
	8.21, //Mn
	8.28, //Fe
	8.35, //Co
	8.41, //Ni
	8.44, //Cu
	8.57, //Zn
	6.32, //Ga
	6.876, //Ge
	7.618, //As
	8.262, //Se
	8.85, //Br
	11.43, //Kr
	3.692, //Rb
	4.88, //Sr
	5.62, //Y
	7.1, //Zr
	6.76, //Nb
	7.51, //Mo
	7.98, //Tc
	8.03, //Ru
	8.01, //Rh
	8, //Pd
	6.268, //Ag
	7.914, //Cd
	5.792, //In
	6.248, //Sn
	6.684, //Sb
	7.052, //Te
	7.524, //I
	9.95, //Xe
	3.422, //Cs
	4.792, //Ba
	5.483, //La
	5.384, //Ce
	5.128, //Pr
	5.241, //Nd
	5.346, //Pm
	5.439, //Sm
	5.575, //Eu
	5.949, //Gd
	5.668, //Tb
	5.743, //Dy
	5.782, //Ho
	5.829, //Er
	5.866, //Tm
	5.93, //Yb
	4.926, //Lu
	6.8, //Hf
	5.7, //Ta
	6.62, //W
	7.84, //Re
	7.26, //Os
	8, //Ir
	8.86, //Pt
	5.172, //Au
	8.32, //Hg
	5.8, //Tl
	7.06, //Pb
	7.48, //Bi
	8.42, //Po
	9.5, //At
	10.74, //Rn
	4, //Fr
	4.868, //Ra
	5.67, //Ac
	5.81, //Th
	5.81, //Pa
	5.706, //U
	5.434, //Np
	5.638, //Pu
	6.007, //Am
	6.007, //Cm
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0 //
};

static const double CHI_UFF[N_ELEMENTS]={
	4.528, //H
	9.66, //He
	3.006, //Li
	4.877, //Be
	5.11, //B
	5.343, //C
	6.899, //N
	8.741, //O
	10.874, //F
	11.04, //Ne
	2.843, //Na
	3.951, //Mg
	4.06, //Al
	4.168, //Si
	5.463, //P
	6.928, //S
	8.564, //Cl
	9.465, //Ar
	2.421, //K
	3.231, //Ca
	3.395, //Sc
	3.47, //Ti
	3.65, //V
	3.415, //Cr
	3.325, //Mn
	3.76, //Fe
	4.105, //Co
	4.465, //Ni
	4.2, //Cu
	5.106, //Zn
	3.641, //Ga
	4.051, //Ge
	5.188, //As
	6.428, //Se
	7.79, //Br
	8.505, //Kr
	2.331, //Rb
	3.024, //Sr
	3.83, //Y
	3.4, //Zr
	3.55, //Nb
	3.465, //Mo
	3.29, //Tc
	3.575, //Ru
	3.975, //Rh
	4.32, //Pd
	4.436, //Ag
	5.034, //Cd
	3.506, //In
	3.987, //Sn
	4.899, //Sb
	5.816, //Te
	6.822, //I
	7.595, //Xe
	2.183, //Cs
	2.814, //Ba
	2.836, //La
	2.774, //Ce
	2.858, //Pr
	2.869, //Nd
	2.881, //Pm
	2.912, //Sm
	2.879, //Eu
	3.167, //Gd
	3.018, //Tb
	3.056, //Dy
	3.127, //Ho
	3.187, //Er
	3.251, //Tm
	3.289, //Yb
	2.963, //Lu
	3.7, //Hf
	5.1, //Ta
	4.63, //W
	3.96, //Re
	5.14, //Os
	5, //Ir
	4.79, //Pt
	4.894, //Au
	6.27, //Hg
	3.2, //Tl
	3.9, //Pb
	4.69, //Bi
	4.21, //Po
	4.75, //At
	5.37, //Rn
	2, //Fr
	2.843, //Ra
	2.835, //Ac
	3.175, //Th
	2.985, //Pa
	3.341, //U
	3.549, //Np
	3.243, //Pu
	2.99, //Am
	2.832, //Cm
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0, //
	0 //
};

static const double ALPHA[N_ELEMENTS]={
	4.50711,//H
	1.38375,//He
	164.1125,//Li
	37.74,//Be
	20.5,//B
	11.3,//C
	7.4,//N
	5.3,//O
	3.74,//F
	2.6611,//Ne
	162.7,//Na
	71.2,//Mg
	57.8,//Al
	37.3,//Si
	25,//P
	19.4,//S
	14.6,//Cl
	11.083,//Ar
	289.7,//K
	160.8,//Ca
	97,//Sc
	100,//Ti
	87,//V
	83,//Cr
	68,//Mn
	62,//Fe
	55,//Co
	49,//Ni
	46.5,//Cu
	38.67,//Zn
	50,//Ga
	40,//Ge
	30,//As
	28.9,//Se
	21,//Br
	16.78,//Kr
	319.8,//Rb
	197.2,//Sr
	162,//Y
	112,//Zr
	98,//Nb
	87,//Mo
	79,//Tc
	72,//Ru
	66,//Rh
	26.14,//Pd
	55,//Ag
	46,//Cd
	65,//In
	53,//Sn
	43,//Sb
	38,//Te
	32.9,//I
	27.32,//Xe
	400.9,//Cs
	272,//Ba
	215,//La
	205,//Ce
	216,//Pr
	208,//Nd
	200,//Pm
	192,//Sm
	184,//Eu
	158,//Gd
	170,//Tb
	163,//Dy
	156,//Ho
	150,//Er
	144,//Tm
	139,//Yb
	137,//Lu
	103,//Hf
	74,//Ta
	68,//W
	62,//Re
	57,//Os
	54,//Ir
	48,//Pt
	36,//Au
	33.91,//Hg
	50,//Tl
	47,//Pb
	48,//Bi
	44,//Po
	42,//At
	45,//Rn
	317.8,//Fr
	246,//Ra
	203,//Ac
	217,//Th
	154,//Pa
	129,//U
	151,//Np
	132,//Pu
	131,//Am
	144,//Cm
	125,//Bk
	122,//Cf
	118,//Es
	113,//Fm
	109,//Md
	110,//No
	320,//Lr
	112,//Rf
	42,//Db
	40,//Sg
	38,//Bh
	36,//Hs
	34,//Mt
	32,//Ds
	32,//Rg
	28,//Cn
	29,//Nh
	31,//Fl
	71,//Mc
	0,//Lv
	76,//Ts
	58//Og
};
//units - Bohr^3
//P. Schwerdtfeger and J.K. Nagle, Molecular Physics 117, 1200 (2019)

//*********************************************
//Function
//*********************************************

//==== NAME ====
const char* name(int an);

//==== ATOMIC_NUMBER ====

int an(const char* name);

//==== MASS ====

double mass(int an);

//==== RADIUS ====

double radius_covalent(int an);
double radius_vdw(int an);

//==== Electric ====

double IE(int an);//ionization energy
double AN(int an);//electron affinity
double CHI(int an);//electronegativity
double ETA(int an);//chemical hardness
double chi_uff(int an);//electronegativity
double eta_uff(int an);//chemical hardness
double alpha(int an);

}

#endif
