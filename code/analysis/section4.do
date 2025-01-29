**************************************************************************************
* Code for Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments
* By Verónica Bäcker-Peral, Jonathon Hazell, and Atif Mian
**************************************************************************************

*******************************
* Table ?: Balance Test
********************************

*******************************
* Table 2: Placebo Test
********************************
use "$clean/renovations_by_experiment.dta", clear

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

use "$clean/experiment_rent_panel.dta", clear
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
use "$clean/rent_rsi.dta", clear
gegen experiment=group(property_id date_trans)

keep experiment property_id year_rm L_year_rm year L_year d_log_rent_res d_rsi_rent_resid date* L_date*
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

use "$clean/rent_rsi.dta", clear
gegen experiment=group(property_id date_trans)
replace d_log_rent = d_log_rent_res 
gen d_rsi = d_rsi_rent_resid

// gen d_rsi = d_rsi_rent

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
