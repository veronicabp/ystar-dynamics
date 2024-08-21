**************************************************************************************
* Code for Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments
* By Verónica Bäcker-Peral, Jonathon Hazell, and Atif Mian
**************************************************************************************

use "$working/experiment_pids.dta", clear
gegen experiment=group(experiment_pid experiment_date)
rename date date_trans

drop if experiment_pid==property_id & type=="control"
gen extension=type=="extension"

merge m:1 property_id date_trans using "$clean/leasehold_flats", keepusing(bedrooms bathrooms livingrooms floorarea age log_rent date_bedrooms date_bathrooms date_livingrooms date_floorarea date_yearbuilt date_rent L_bedrooms L_bathrooms L_livingrooms L_floorarea L_log_rent L_date_bedrooms L_date_bathrooms L_date_livingrooms L_date_floorarea L_date_yearbuilt L_date_rent L_date_trans) nogen keep(match)

merge m:1 property_id date_trans using "$working/renovations", keepusing(date_rm renovated) nogen keep(master match)

rename (property_id date_trans) (property_id_c date_trans_c)
rename (experiment_pid experiment_date) (property_id date_trans)

merge m:1 property_id date_trans using "$clean/experiments", keepusing(k90 date_extended) nogen keep(match)

label var floorarea "Floor Area"
label var log_rent "Log Rent"
label var bathrooms "Bathrooms"
label var bedrooms "Bedrooms"
label var livingrooms "Living Rooms"
label var age "Property Age"

************************************
* Appendix Figure ??: Density plots 
************************************

gen date_age = date_yearbuilt
foreach var in bedrooms bathrooms livingrooms floorarea age rent {
	gen year_`var' = year(date_`var')
}

foreach var of varlist bedrooms bathrooms livingrooms floorarea age log_rent {
		preserve
		drop if missing(`var')
		
		cap drop `var'_res
		
		qui: gen temp=`var' if extension 
		qui: gegen temp=mean(temp), by(experiment) replace 
		qui: drop if missing(temp)
		qui: drop temp
		
		qui: gen temp=`var' if !extension 
		qui: gegen temp=mean(temp), by(experiment) replace 
		qui: drop if missing(temp)
		qui: drop temp
		
		count
		
		if "`var'" == "log_rent" qui reghdfe `var' , absorb(i.experiment#i.year_rent) residuals(`var'_res)
		else qui reghdfe `var', absorb(i.experiment#i.year_`var') residuals(`var'_res)
		
		local lab: variable label `var'
		local title "`lab', Residualized"
		
		twoway (kdensity `var'_res if extension, bwidth(1) kernel(gaussian) lcolor("$accent1")) ///
			 (kdensity `var'_res if !extension, bwidth(1) kernel(gaussian) lcolor(gs4) lpattern(dash)), ///
			ytitle("Density") ///
			xtitle("`title'") ///
			legend(order(1 "Extended" 2 "Not Extended"))
		graph export "$fig/`var'_kdensity.png", replace  
	restore
}

*******************************
* Table 2: Placebo Test
********************************
foreach var of varlist bedrooms bathrooms livingrooms floorarea {
	gen d_`var' = `var' - L_`var' if date_`var' > date_extended & date_extended > L_date_`var' & `var'>=L_`var' 
}

gen n=_n
gcollapse d_* renovated (count) n n_bed=d_bedrooms, by(experiment extension)

eststo clear
foreach var of varlist renovated d_* {
	di "`var'"
	qui eststo: reghdfe `var' i.extension, absorb(i.experiment) vce(robust)
	qui: estadd local control_mean = round(_b[_cons]*1000)/1000
	qui: estadd local fe "\checkmark"
	qui: distinct experiment if e(sample)
	qui: estadd local exp_count = string(r(ndistinct), "%9.0fc"), replace
}

esttab using "$tab/placebo_test.tex", ///
	keep(1.extension) varlabel(1.extension "Extension") ///
	s(fe control_mean N exp_count, label("Experiment FE" "Control Mean" "N" "N. Experiment") fmt("%9.0fc")) ///
	mtitles("Renovation Rate" "$\Delta$ Bedrooms" "$\Delta$ Bathrooms" "$\Delta$ Living Rooms" "$\Delta$ Floor Area") ///
	se replace se(3) b(3) 

*************************************
* Table 3: Rent Growth
*************************************

use "$working/experiment_rent_panel.dta", clear
replace d_log_rent = d_log_rent_res
replace d_log_rent_ann = d_log_rent_res /((date_rm-L_date_rm)/365)
keep if  year_rm<=year & L_year_rm>=L_year // Keep rental listings within experiment window

gen control = type=="control"
gen extension = type=="extension"

eststo clear
eststo: reghdfe d_log_rent extension, absorb(experiment##year_rm##L_year_rm) vce(robust)
estadd local fe1 "\checkmark"
distinct experiment if e(sample) & extension
estadd local exp_count = string(r(ndistinct), "%9.0fc"), replace

gcollapse d_log_rent_ann, by(experiment extension)
eststo: reghdfe d_log_rent_ann extension , absorb(experiment) vce(robust)
estadd local fe2 "\checkmark"
estadd local annual "\checkmark"
distinct experiment if e(sample)
estadd local exp_count = string(r(ndistinct), "%9.0fc"), replace

// RSI version
import delimited "$working/rsi/rsi_rent_resid.csv", clear case(preserve)
replace d_log_rent = d_log_rent_res
gen date_rm_ = date(date_rm, "DMY")
gen L_date_rm_ = date(L_date_rm, "DMY")
drop date_rm L_date_rm 
rename (date_rm_ L_date_rm_) (date_rm L_date_rm )
 
drop date_extended
joinby property_id using "$clean/experiments.dta"
gegen experiment=group(property_id date_trans)

keep experiment property_id year_rm L_year_rm year L_year d_log_rent d_rsi date_rm L_date_rm date_trans L_date_trans
rename d_log_rent d_log_rentextension
rename d_rsi d_log_rentcontrol

// Remove outliers as in main analysis
gen did = d_log_rentextension   - d_log_rentcontrol 
winsor did, p(0.005) gen(w)
keep if w==did

reshape long d_log_rent, i(experiment year_rm L_year_rm) j(type) string
gen extension = type=="extension"
eststo: reghdfe d_log_rent extension if  year_rm<=year & L_year_rm>=L_year, absorb(experiment#year_rm#L_year_rm) vce(roubst)
estadd local fe1 "\checkmark"
estadd local rsi "\checkmark"

distinct experiment if e(sample)
estadd local exp_count = string(r(ndistinct), "%9.0fc"), replace

esttab using "$tab/within_experiment_rent_growth.tex", ///
	mgroups("$ \Delta$ log(Rent)", pattern(1 ) ///
			prefix(\multicolumn{@span}{c}{) suffix(}) ///
			span erepeat(\cmidrule(lr){@span})) ///
	nomtitle b(4) se(4) ///
	keep(extension) varlabel(extension "Extension") ///
	stats(fe1 fe2 annual rsi N exp_count, label("Experiment $\times$ Rent Years FE" "Experiment FE" "Annualized" "RSI" "N" "N. Experiment") fmt(%9.0fc)) replace

*************************************
* Figure 5: Rent Growth
*************************************

import delimited "$working/rsi/rsi_rent.csv", clear case(preserve)
replace d_log_rent = d_log_rent_res

drop date_extended
joinby property_id using "$clean/experiments.dta"

gegen experiment=group(property_id date_trans)

// Remove outliers as in main analysis
gen did = d_log_rent - d_rsi 
winsor did, p(0.005) gen(w)
keep if w==did

// Add an extra observation before the first to initialize
gsort experiment year_rm
by experiment: gen first=_n==1
expand 2 if first, gen(idx)

gsort experiment year_rm -idx
replace year_rm = L_year_rm if idx==1 

// Create a treated index 
gen log_rent_ext = 0 if idx==1 
by experiment: replace log_rent_ext = log_rent_ext[_n-1] + d_log_rent if _n>1

// Create a control index 
gen log_rent_ctrl = 0 if idx==1 
by experiment: replace log_rent_ctrl = log_rent_ctrl[_n-1] + d_rsi if _n>1

gen mid_year = round( (year + L_year)/2 )
gen time = year_rm - mid_year

keep experiment property_id time log_rent_ext log_rent_ctrl
xtset experiment time

tsfill
gsort experiment -property_id
foreach var of varlist property_id { //date_trans L_date_trans year L_year year_rm L_year_rm
	by experiment: replace `var' = `var'[1]
}
gsort experiment time

ipolate log_rent_ext time, gen(log_rent_ext_) by(experiment)
gen d_log_rent_ext = log_rent_ext_ - L.log_rent_ext_

ipolate log_rent_ctrl time, gen(log_rent_ctrl_) by(experiment)
gen d_log_rent_ctrl = log_rent_ctrl_ - L.log_rent_ctrl_

drop if missing(d_log_rent_ext) | missing(d_log_rent_ctrl)
gcollapse d_log_rent_ext d_log_rent_ctrl (semean) se_ext=d_log_rent_ext se_ctrl = d_log_rent_ctrl, by(time)

foreach x in "_ext" "_ctrl" {
	gen ub`x' = d_log_rent`x' + 1.96*se`x'
	gen lb`x' = d_log_rent`x' - 1.96*se`x'
}

twoway 	(scatter d_log_rent_ext time, mcolor("$accent1") msymbol(S)) ///
		(scatter d_log_rent_ctrl time, mcolor(gs2) msymbol(D)) ///
		(rcap ub_ext lb_ext time, lcolor("$accent1")) ///
		(rcap ub_ctrl lb_ctrl time, lcolor(gs2%50)) if se_ctrl<0.005 & se_ext<0.005, ///
		xtitle("Years Since Experiment Mid-Point") ytitle("∆ Log(Rent)") ///
		legend(order(1 "Extension" 2 "Control") ring(0) position(11)) ///
		xline(0, lcolor(black) lpattern(dash)) ylabel(0(0.01)0.05)
graph export "$fig/rent_growth_horizons_rsi.png", replace
