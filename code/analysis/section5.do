**************************************************************************************
* Code for Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments
* By Verónica Bäcker-Peral, Jonathon Hazell, and Atif Mian
**************************************************************************************

*************************************
* Figure 6: Event Study
*************************************
use "$clean/ystar_estimates.dta", clear
keep if freq=="annual" | freq=="2000-2003"
rename date year
keep year ystar
tempfile ystar_yearly
save `ystar_yearly'

use "$clean/for_event_study.dta", clear
// keep if round(extension_amount, 5)==90

// Set level to 0 in year before extension
gsort property_id date_trans 
by property_id: gen L_rsi = rsi[_n-1] 
gen d_rsi = (rsi - L_rsi) + constant

drop if years_held<2 // Drop flippers
by property_id: gen F_extension = extension[_n+1] 

keep if extension==1 | F_extension==1

gen time = datediff(date_extended, date_trans, "year") 
drop if time<0 & extension==1 // Drop extensions that were marked as extended before purchase
drop if time>0 & F_extension==1 // Drop extensions that were marked as extended after sale

replace time=time+0.5 if extension==1 
replace time=time-0.5 if F_extension==1

gen did = d_log_price - d_rsi 
gcollapse diff=did T=experiment_duration k=extension_amount year=experiment_year (semean) se=did, by(time)

gen ub = diff + 1.96*se
gen lb = diff - 1.96*se

replace year=round(year)
merge m:1 year using `ystar_yearly', nogen keep(match)
gen pred_y =  ln(1-exp(-(ystar/100) *(T + k))) - ln(1-exp(-(ystar/100) *(T)))

gsort time
twoway 	(scatter diff time if time<0) ///
		(rcap ub lb time if time<0) ///
		(scatter diff time if time>0, mcolor("$accent1") msymbol(D)) ///
		(rcap ub lb time if time>0, lcolor("$accent1")) ///
		(line pred_y time if time>0, lpattern(dash) lcolor(gs10)) if abs(time)<=15, ///
		xtitle("Years Since Extension") ytitle("∆ Market Value for Extended Properties" "(Relative to Controls)") ///
		legend(order(1 "Pre-Extension ∆ Market Value " 3 "Post-Extension ∆ Market Value" 5 "Predicted Trend Under Gordon Growth Formula") ring(0) position(11))  ///
		xlabel(-15(5)15) ylabel(-0.1(0.1)0.3) xline(0, lcolor("gs5") lpattern(dash)) yline(0, lcolor("gs5") lpattern(dash))
graph export "$fig/event_study_main.png", replace

*************************************
* Figure 7: Binscatter
*************************************

use "$clean/experiments.dta", clear

foreach sample of varlist k90 k700p {
	preserve 
		keep if `sample' & T<100
		
		nl $nlfunc, initial(ystar 3) variables(did_rsi T k)
		global ystar = _b[/ystar]
		
		if "`sample'" == "k90" local k=90
		else local k=900
	
		binscatter2 did_rsi T, nq(100) savedata("$working/binned") replace
		import delimited "$working/binned.csv", clear case(preserve)
		drop __000001

		twoway (scatter did_rsi T if T<=120, mcolor("$accent1") msymbol(Oh)) ///
			   (function did_rsi = ( ln(1-exp(-$ystar /100 * (x+`k'))) - ln(1-exp(-$ystar /100 * x)) ) , ///
					range(30 120) lcolor(black) lwidth(medthick)), ///
			   legend(order(2 "Predicted Values From Asset Pricing Function") ring(0) position(2)) xlabel(30(30)120) ///
			   xtitle("Duration Before Extension") ///
			   ytitle("∆ Market Value for Extended Properties" "(Relative to Controls)") 
		graph export "$fig/yield_curve_`sample'.png", replace
	restore
}

*************************************
* Figure 8: Yield Curve Dynamics
*************************************

use "$clean/experiments.dta", clear
keep if k90

cap drop period
gen period = 1 if year<=2008
replace period = 2 if year > 2008 & year <= 2016
replace period = 3 if year > 2016 

forv p=1/3 {
	qui: nl $nlfunc if period==`p', initial(ystar 3) variables(did_rsi T k)
	global ystar`p' = _b[/ystar]
}

collapse did_rsi (count) n=did_rsi, by(T10 period)
local gap = 2
replace T = T-`gap' if period==1
replace T = T+`gap' if period==3

local gap=2
local gr ""
forv p=1/3 {
	if `p'==1 local color "dkorange" 
	if `p'==2 local color "midblue" 
	if `p'==3 local color "lavender" 
	
	qui: sum n if period==`p'
	local num_obs=r(max)
	
	forv i=0(10)`num_obs' {
		local pct = round(100 * `i'/(`num_obs')) + 5 
		if `pct'>100 local pct = 100
		local gr "`gr' (bar did_rsi T if period==`p' & n>=`i' & n<`i'+10, bcolor(`color'%`pct') barwidth(`gap'))"
	}
}
twoway	(function y=ln(1-exp(-($ystar1 /100) * ((x+`gap')+90) )) - ln(1-exp(-($ystar1 /100) * (x+`gap'))), ///
			range(30 100) lpattern(dash) lcolor(dkorange)) ///
		(function y=ln(1-exp(-($ystar2 /100) * (x+90) )) - ln(1-exp(-($ystar2 /100) * x)), ///
			range(30 100) lpattern(dash) lcolor(midblue)) ///
		(function y=ln(1-exp(-($ystar3 /100) * ((x-`gap')+90) )) - ln(1-exp(-($ystar3 /100) * (x-`gap'))), ///
			range(30 100) lpattern(dash) lcolor(lavender)) ///
		`gr' ///
		if T>=40 & T<=100 & did>=0 & did<=0.7, ///
		xlabel(40(10)100) ///
		legend(order(1 "Pre 2008" 2 "2008-2016" 3 "Post 2016")) ///
		xtitle("Duration") ytitle("∆ Market Value for Extended Properties" "(Relative to Controls)") 
graph export "$fig/yield_curve_3period_k90.png", replace

*************************************
* Figure 9: Real-time estimates
*************************************

use "$clean_update/ystar_monthly_estimates.dta", clear
merge 1:m year month using "$clean/uk_interest_rates.dta", keep(match master)
gen xaxis = year + (month-1)/12

// With forward rate
twoway 	(line 		ystar xaxis									, yaxis(1) lpattern(solid) lcolor(gs10)) ///
		(rarea 		ub lb xaxis if _n!=_N, yaxis(1) color(gs10%30) lcolor(%0)) ///
		(scatter 	ystar xaxis									, yaxis(1) mcolor(black) msymbol(O)) ///
		(scatter 	ystar xaxis if _n==_N	, yaxis(1) mcolor("$accent1") msymbol(O)) ///
		(line 		uk10y20_real xaxis										, yaxis(2) lcolor(gs4) lpattern(shortdash)) , ///
		legend(order(3 "y* (Left Axis)" 5 "10 Year 20 Real Forward Rate (Right Axis)") ring(0) position(11)) ///
		xtitle("") ytitle("", axis(1)) ytitle("", axis(2)) ///
		ylabel(2(1)7, axis(1)) ylabel(-2.5(1)2.5, axis(2)) ///
		xlabel(2016 "2016, Q1" 2018 "2018, Q1" 2020 "2020, Q1" 2022 "2022, Q1" 2024 "2024, Q1" 2025.6 " ") ///
		text(2.35 2025.2 "$month1_str" "$year1", color("$accent1") size(smal))
graph export "$fig/realtime_updates_with_forward_monthly.png", replace

*************************************
* Figure 10: y* stability
*************************************

use "$clean/hedonics_variations.dta", clear

gen special = inlist(variation, "None", "linear", "all", "quad", "all_gms")
replace ystar_qe = . if variation=="all_gms"

foreach tag in "cs" "qe" {
	egen ystar_`tag'_max = max(ystar_`tag'), by(time_interaction)
	egen ystar_`tag'_min = min(ystar_`tag'), by(time_interaction)
	
	* Make the max/min line go to the end of the dot
	replace ystar_`tag'_max = ystar_`tag'_max + 0.09
	replace ystar_`tag'_min = ystar_`tag'_min - 0.09
}

gen gms_estimate = 1.9 if _n==1

gen yaxis_cs = 1
gen yaxis_qe = 0.75
gen yaxis_gms_estimate = 1.025

twoway 	(scatter yaxis_cs ystar_cs if !special, mcolor(gs4%5) msymbol(o) msize(huge) lcolor(black)) ///
		(scatter yaxis_qe ystar_qe if !special, mcolor(gs4%1) msymbol(o) msize(huge) lcolor(black)) ///
		(rcap ystar_cs_max ystar_cs_min yaxis_cs, lcolor(black) lpattern(dash) horizontal) ///
		(rcap ystar_qe_max ystar_qe_min yaxis_qe, lcolor(black) lpattern(solid) horizontal) ///
		(scatter yaxis_cs ystar_cs if variation=="quad", mcolor("0 150 255") msymbol(O) msize(large) mlcolor(black)) ///
		(scatter yaxis_cs ystar_cs if variation=="linear", mcolor("0 200 255") msymbol(O) msize(large) mlcolor(black)) ///
		(scatter yaxis_cs ystar_cs if variation=="None", mcolor("0 255 255") msymbol(O) msize(large) mlcolor(black)) ///	
		(scatter yaxis_cs ystar_cs if variation=="all", mcolor("0 0 255") msymbol(O) msize(large) mlcolor(black)) ///
		(scatter yaxis_qe ystar_qe if variation=="quad", mcolor("0 150 255") msymbol(O) msize(small) mlcolor(black)) ///
		(scatter yaxis_qe ystar_qe if variation=="linear", mcolor("0 200 255") msymbol(O) msize(small) mlcolor(black)) ///
		(scatter yaxis_qe ystar_qe if variation=="None", mcolor("0 255 255") msymbol(O) msize(small) mlcolor(black)) ///	
		(scatter yaxis_qe ystar_qe if variation=="all", mcolor("0 0 255") msymbol(O) msize(small) mlcolor(black) mlcolor(black)) ///
		(scatter yaxis_gms_estimate gms_estimate, msymbol(S) mcolor(black)  msize(medium)) ///
		(scatter yaxis_gms_estimate gms_estimate, msymbol(X) mcolor(gold)  msize(medium)) ///
		(scatter yaxis_cs ystar_cs if variation=="all_gms", msymbol(S) mcolor(black)  msize(medium)) ///
		(scatter yaxis_cs ystar_cs if variation=="all_gms", mcolor(red) msymbol(X) msize(medium)) if time_interaction==0, ///
		legend(order(7 "No Controls" 6 "Linear" 5 "Quadratic" 8 "Fixed Effects") position(0) bplacement(seast) cols(1) size(*0.8)) ///
		yscale(range(0.6(0.1)1.2)) ///
		xtitle("Estimated y*") ytitle("") ///
		ylabel(0.5 " " 0.75 `""Quasi" "Experimental""' 1 `""Cross" " Sectional""' 1.25 " ") ////
		xlabel(0(1)10)  ///
		text(1.065 1.9 "Giglio, Maggiori & Stroebel (2015)" "Published Results", color(gold*1.5) size(vsmall)) ///
		text(.94 2.3 "Giglio, Maggiori & Stroebel (2015)" "Replication On Our Sample", color(cranberry*1.5) size(vsmall))
graph export "$fig/ystar_stability.png", replace


twoway 	(scatter yaxis_cs ystar_cs, mcolor(gs4%5) msymbol(o) msize(huge) lcolor(black)) ///
		(scatter yaxis_qe ystar_qe, mcolor(gs4%1) msymbol(o) msize(huge) lcolor(black)) ///
		(rcap ystar_cs_max ystar_cs_min yaxis_cs, lcolor(black) lpattern(dash) horizontal) ///
		(rcap ystar_qe_max ystar_qe_min yaxis_qe, lcolor(black) lpattern(solid) horizontal) if time_interaction==1, ///
		legend(off) ///
		yscale(range(0.6(0.1)1.2)) ///
		xtitle("Estimated y*") ytitle("") ///
		ylabel(0.5 " " 0.75 `""Quasi" "Experimental""' 1 `""Cross" " Sectional""' 1.25 " ") ////
		xlabel(0(1)6)
graph export "$fig/ystar_stability_yearfe.png", replace

*************************************
* Table: y* Estimates
*************************************

use "$clean/experiments.dta", clear
gegen lpa_code_n = group(lpa_code)

reghdfe did_rsi, absorb(year#lpa_code_n) residuals(res)
sum did_rsi 
gen did_res = r(mean) + res

estimates clear

// 90 year extensions 
nl (did_res = ln(1-exp(-({ystar=3}/100)*(T+k))) - ln(1-exp(-({ystar=3}/100)*T))) if k90, vce(robust)
local ystar_k90=_b[/ystar]
local ystar_k90_se=_se[/ystar]
store_nlls_estimates "ystar" "k90"
estadd local obs =  string(e(N), "%10.0gc")

* Flexible yield curve
local ystar_param "{ystar} + {b1}*(T)"
nl (did_res = ln(1-exp(-((`ystar_param')/100)*(T+k))) - ///
		ln(1-exp(-((`ystar_param')/100)*T))) if k90, ///
	initial(ystar 0.1 b1 0.1) variables(did_rsi T) vce(robust)
	
// For multiple points along the yield curve, calculate implied ystar + standard error
forv T = 50(10)80 {
	// Store ystar(T)
	local b`T' = _b[/ystar] + _b[/b1] * (`T')
	local v`T' = e(V)[1,1] + 2*e(V)[1,2]*(`T') + e(V)[2,2]*(`T')^2
}

// Store estimates
forv T = 50(10)80 {
	matrix b = (`b`T'')
	matrix V = (0) //  `v`T''
	
	matrix list b 
	matrix list V

	matrix coleq b = " "
	matrix coleq V = " "

	matrix colnames b = "ystar"
	matrix colnames V = "ystar"

	qui: regress did_rsi T, nocons
	erepost b = b V = V, rename
	estimates store T`T'
}

// 700+ year extensions 
nl (did_res = ln(1-exp(-({ystar=3}/100)*(T+k))) - ln(1-exp(-({ystar=3}/100)*T))) if k700p, vce(robust)
local ystar_k700p=_b[/ystar]
local ystar_k700p_se=_se[/ystar]
store_nlls_estimates "ystar" "k700p"
estadd local obs =  string(e(N), "%10.0gc")

// T-test
local ttest = (`ystar_k700p' - `ystar_k90')/sqrt((`ystar_k700p_se')^2 + (`ystar_k90_se')^2)
estadd local ttest = round(`ttest'*100)/100

esttab * using "$tab/ystar.tex", /// 
	mgroups("Constant $ y^*$" "Flexible $ y^*$" "Constant $ y^*$", pattern(1 1 0 0 0 1) ///
		prefix(\multicolumn{@span}{c}{) suffix(}) ///
		span erepeat(\cmidrule(lr){@span})) ///
	mtitle("$ k=90$" "$ T=50$" "$ T=60$" "$ T=70$" "$ T=80$" "$ k\geq 700$") ///
	varlabel(ystar "$ y^*$") ///
	se replace b(2) se(3) stats(obs ttest, label("N" "t-stat (700+ vs. 90)")) substitute("(.)" "") 
