**************************************************************************************
* Code for Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments
* By Verónica Bäcker-Peral, Jonathon Hazell, and Atif Mian
**************************************************************************************

*************************************
* Table 1: Extension Frequency
*************************************

use "$clean/experiments.dta", clear

gen extension_bin = 1 if round(extension_amount,5)==90
replace extension_bin = 2 if extension_amount > 700
replace extension_bin = 3 if extension_bin == .

gen year_bin = "2000-2005" if year>=2000 & year<=2005
replace year_bin = "2006-2010" if year>2005 & year<=2010
replace year_bin = "2011-2015" if year>2010 & year<=2015
replace year_bin = "2016-2020" if year>2015 & year<=2020
replace year_bin = "2021-2023" if year>2020

eststo clear
estpost tabulate year_bin extension_bin

esttab using "$tab/extension_count.tex", ///
		cell(b(fmt(%12.0gc))) unstack collabels(none) noobs nonumber nomtitle  ///
		varlabels(, blist(Total "\hline ")) ///
		eqlabels("90" "700+" "Other", lhs("Extension Amount")) replace

*******************************************
* Appendix Table: Extension Frequency Full
*******************************************

use "$clean/leasehold_flats.dta", clear

// Set of current and future experiments
keep if has_been_extended | extension
gsort property_id -extension
gduplicates drop property_id, force

gen extension_bin = 1 if round(extension_amount,5)==90
replace extension_bin = 2 if extension_amount > 700
replace extension_bin = 3 if extension_bin == .

gen year_bin = "2000-2005" if year_extended>=2000 & year_extended<=2005
replace year_bin = "2006-2010" if year_extended>2005 & year_extended<=2010
replace year_bin = "2011-2015" if year_extended>2010 & year_extended<=2015
replace year_bin = "2016-2020" if year_extended>2015 & year_extended<=2020
replace year_bin = "2021-2023" if year_extended>2020

eststo clear
estpost tabulate year_bin extension_bin

esttab using "$tab/extension_count_full.tex", ///
		cell(b(fmt(%12.0gc))) unstack collabels(none) noobs nonumber nomtitle  ///
		varlabels(, blist(Total "\hline ")) ///
		eqlabels("90" "700+" "Other", lhs("Extension Amount")) replace
