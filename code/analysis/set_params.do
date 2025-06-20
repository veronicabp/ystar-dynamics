
// Set paths
global folder "/Users/vbp/Dropbox (Personal)/research/github-projects/ystar-dynamics"
global dropbox "/Users/vbp/Dropbox (Personal)/"
global data_folder "$folder/data/data"

global clean "$data_folder/original/clean"
global working "$data_folder/original/working"
global raw "$data_folder/original/raw"

global update_folder "update_03_12_24"
global clean_update "$data_folder/$update_folder/clean"


global data "$folder/data/clean"
global overleaf "$dropbox/Apps/Overleaf/UK Duration"
global fig "$overleaf/Figures"
global tab "$overleaf/Tables"

// Declare parameters
global hedonics_rm "bedrooms floorarea bathrooms livingrooms yearbuilt"
global hedonics_rm_full "$hedonics_rm parking currentenergyrating heatingtype condition"
global hedonics_zoop "bathrooms_zoop bedrooms_zoop floors receptions"

global year0 = 2003 
// global year1 = 2023
// global month1 = 12
// global month1_str = "December"
global year1 = 2024
global month1 = 10
global month1_str = "October"

global ystar_func "({ystar}/100)"
global nlfunc "(did_rsi = ln(1-exp(- $ystar_func * (T+k))) - ln(1-exp(-$ystar_func * T)))"

// Set accent colors 
global accent1 "7 179 164"
global accent2 "88 40 209"
global accent3 "209 147 31"
global accent1_dark "8 99 92"

set scheme plotplainblind
graph set window fontface "Times New Roman" 

adopath ++ "$folder/code/ado"
