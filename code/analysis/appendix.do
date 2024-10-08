**************************************************************************************
* Code for Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments
* By Verónica Bäcker-Peral, Jonathon Hazell, and Atif Mian
**************************************************************************************

*****************************************
* Figure A.1: Lease Term Histogram
*****************************************

use "$clean/leasehold_flats.dta", clear
histogram duration if duration<1000, xtitle("Duration") percent width(5)
graph export "$fig/sale_duration_histogram.png", replace

*************************************
* Figure A.2: Rennovations
*************************************

use "$clean/leasehold_flats.dta", clear
cap gen years_held_n = round(years_held)
binscatter2 d_bedrooms years_held_n if date_bedrooms!=L_date_bedrooms & d_bedrooms>=0 & years_held < 10, xtitle("Years Held") ytitle("Mean Change in Number of Bedrooms") line(connect) msymbol(O) 
graph export "$fig/d_bedrooms_by_years_held.png", replace

use "$working/renovations.dta", clear
cap gen years_held_n = round(years_held)
binscatter2 renovated years_held_n if years_held < 10, xtitle("Years Held") ytitle("Renovation Rate") line(connect) msymbol(O) 
graph export "$fig/renovations_by_years_held.png", replace

*************************************
* Figure A.3: Hazard Rate
*************************************

use "$clean/leasehold_flats.dta", clear  
* Infer how many extensions we are missing:
distinct property_id if closed_lease & number_years==99 & round(extension_amount)==90 // Total number of recorded 189 extensions with pre-transaction data
local tot = r(ndistinct)
distinct property_id if number_years == 189 & L_date_trans==. // 189 extensions that were not recorded
local missing_pre = r(ndistinct)
global hazard_correction = 1 + `missing_pre'/`tot'

use "$clean/leasehold_panel.dta", clear

keep if L_duration<=125
collapse (sum) num_extensions=extended (count) n=extended, by(L_duration_bin)

gen hazard_rate = 100 * num_extension/n
gen hazard_rate_corrected = hazard_rate * $hazard_correction

foreach tag in "" "_corrected" {
	gen se`tag' = 100 * sqrt( (hazard_rate`tag'/100) *(1 - (hazard_rate`tag'/100))/n)

	gen ub`tag' = hazard_rate`tag' + 1.96*se`tag'
	gen lb`tag' = hazard_rate`tag' - 1.96*se`tag'

	twoway 	(line hazard_rate`tag' L_duration, lcolor(black)) ///
			(rarea ub`tag' lb`tag' L_duration, color(gray%30) lcolor(%0)) if se`tag'>0, ///
			xtitle("Duration") ytitle("% Extended") ///
			xline(80, lcolor(red) lpattern(dash)) ///
			legend(off)  ylabel(0(2)6)
	graph export "$fig/hazard_rate`tag'.png", replace	
}

*************************************
* Figure A.4: Cumulative Hazard Rate
*************************************

use "$clean/leasehold_panel.dta", clear
keep if year>=2003 & year<2020
keep if L_duration<=125 & L_duration > 0
collapse (sum) num_extensions=extended (count) n=extended, by(L_duration)

gen hazard_rate = num_extension/n * $hazard_correction
replace hazard_rate=0 if L_duration>125

* Baseline
gsort -L_duration
gen inv_hazard = 1-hazard
gen prod_inv_hazard = inv_hazard if _n==1
replace prod_inv_hazard = inv_hazard * prod_inv_hazard[_n-1] if _n>1
gen L_prod_inv_hazard = prod_inv_hazard[_n-1]
gen prob_E_AND_T = hazard * L_prod_inv_hazard
gen cum_prob = sum(prob_E_AND_T)
replace cum_prob = cum_prob * 100

twoway (line cum_prob L_duration, lcolor(black)), ///
		xtitle("Duration") ytitle("% Have Extended") xlabel(0(40)120)
graph export "$fig/cumulative_hazard.png", replace

******************************************************
* Figure A.5: Duration Before Extension Histogram
******************************************************
use "$clean/experiments.dta", clear

histogram T_at_ext if T_at_ext<125, xtitle("Duration Before Extension") percent
graph export "$fig/extension_duration_histogram.png", replace

******************************************************
* Figure A.6: Holding Period Historgram
******************************************************

use "$clean/experiments_incflippers.dta", clear
histogram years_held, xtitle("Years Between Transactions") percent xline(2, lpattern(dash) lcolor("$accent1")) width(1)
graph export "$fig/years_held_histogram.png", replace

******************************************************
* Figure A.7: Years To Extension
******************************************************

use "$clean/experiments.dta", clear
gen time_between = datediff(L_date_trans, date_registered, "year")
histogram time_between if time_between<25, xtitle("Years Between Transaction And Extension") percent width(1)
graph export "$fig/time_to_extension_histgram.png", replace

*************************************
* Figure A.8: Rent to Price
*************************************
use "$clean/flats.dta", clear

// Keep properties that have rental data within one year of the transaction data
keep if abs(datediff(date_rent, date_trans, "year"))<=1
qui: gen rtp = 100 * rent/price
qui: reghdfe rtp, absorb(lpa_code) residuals(res) // Residualize on postcode area 
qui: sum rtp 
qui: replace rtp = res + r(mean)

qui: gen ext = has_extension | has_been_extended
gen date = year

qui: gen group = 1 if leasehold & ext 
qui: replace group = 2 if leasehold & !ext 
qui: replace group = 3 if freehold

gcollapse rtp (semean) se=rtp, by(date group)
twoway  (line rtp date if group==1, lcolor("$accent1")) ///
		(line rtp date if group==2, lcolor("$accent2")) ///
		(line rtp date if group==3, lcolor("$accent3")) ///
				if date>=2006 & date<=2022, ///
		legend(order(	1 "Leasehold (Extender)" ///
						2 "Leasehold (Non-Extender)" ///
						3 "Freehold") ///
				ring(0) position(1)) ///
		xtitle("") ytitle("Rent to Price Ratio, Residualied by Local Authority") xlabel(2006(3)2021) ylabel(4.5(1)7.5)
graph export "$fig/rent_to_price_timeseries.png", replace

*******************************************
* Figure A.10: Time From Listing Histogram
*******************************************

use "$working/merged_hmlr.dta", clear
keep if flat
joinby property_id using  "$working/rightmove_merge_keys"
joinby property_id_rm using "$working/rightmove_sales_flats.dta"

// Get listing that most closely precedes transaction
keep if date_trans>date_rm
gen d=date_trans - date_rm
gegen mind=min(d), by(property_id date_trans)
keep if d==mind 
gduplicates drop property_id date_trans, force

histogram d if d < 800 & d>0, xtitle("Days Between Last Listing and Transaction Date") percent width(7)
graph export "$fig/time_from_listing_histogram.png", replace

***************************************** 
* Figure A.11: Holding Period Binscatter
*****************************************

use "$clean/experiments.dta", clear
binscatter2 did_rsi years_held if k90, nq(1000) xtitle("Years Held") ytitle("Price Difference After Extension vs. Control") absorb(year)
graph export "$fig/holding_period_binscatter.png", replace

*****************************************
* Figure A.12: RSI Residuals
*****************************************

import delimited "$working/rsi/rsi_residuals.csv", clear
gen res2 = residuals^2
binscatter2 res2 years_held, xtitle("Years Held") ytitle("Squared Residuals")
graph export "$fig/residuals_years_held_binscatter.png", replace

binscatter2 res2 distance, xtitle("Distance From Treated Property") ytitle("Squared Residuals")
graph export "$fig/residuals_distance_binscatter.png", replace

*****************************************
* Figure A.13: Hedonics Binscatters
*****************************************

use "$clean/leasehold_flats.dta", clear
foreach var of varlist bedrooms bathrooms livingrooms floorarea age log_rent {
	local lab: variable label `var'
	binscatter2 log_price `var', nq(1000) xtitle("`lab', Residualized") ytitle("Log(Price), Residualized") absorb(lpa_code)
	graph export "$fig/log_price_`var'_binscatter.png", replace
}

*****************************************
* Figure A.16: Histogram of Duration
*****************************************

use "$clean/experiments.dta", clear
histogram whb_duration if k90 & whb_duration<160, color(gs4%70) percent width(5) start(0) ///
addplot(histogram duration if k90 & duration<250, color("$accent1"%70) percent width(5) start(0)) ///
 xtitle("Duration") legend(order(1 "Control" 2 "Extended"))
graph export "$fig/duration_histogram.png", replace

*****************************************
* Figure A.19: Post-Extension Trend
*****************************************

use "$clean/experiments.dta", clear
keep if year>=2018
keep if k90

gsort property_id date_trans
gen time = datediff(date_extended, date_trans, "year") + 0.5
keep if time>=0 & time<=10.5

nl $nlfunc, initial(ystar 3)
global ystar = _b[/ystar]/100

gcollapse diff=did_rsi T k (semean) se=did_rsi, by(time)
gen ub = diff + 1.96*se
gen lb = diff - 1.96*se

gen pred_y =  ln(1-exp(-$ystar *(T + k))) - ln(1-exp(-$ystar *(T)))

// Add 0 before extension time 
expand 12 if _n==1, gen(idx)
gsort time -idx
replace time = -_n + 0.5 if idx==1 
replace diff = 0 if idx==1 
replace ub = 0 if idx==1 
replace lb = 0 if idx==1 
replace pred_y = . if idx==1 
gsort time

twoway 	(scatter diff time) ///
		(rcap ub lb time) ///
		(line pred_y time, lpattern(dash) lcolor("$accent1")), ///
		xtitle("Years Since Extension") ytitle("Treated - Control") ///
		legend(order(1 "Observed" 3 "Predicted Assuming Constant Discount Rate") ring(0) position(1))  ///
		xlabel(-10(2)10) ylabel(-0.1(0.1)0.3) xline(0, lcolor("gs5") lpattern(dash)) yline(0, lcolor("gs5") lpattern(dash))
graph export "$fig/post_trend_k90_2018t2023.png", replace

*****

foreach tag in "" "_k90" {
	foreach restrict in "" "_post2012" "_pre2012" "_o70" "_u70" {
		use "$working/for_event_study.dta", clear

		if "`tag'"=="_k90" keep if round(extension_amount, 5)==90

		if "`restrict'"=="_post2012" keep if experiment_year>2012
		if "`restrict'"=="_pre2012" keep if experiment_year<=2012

		if "`restrict'"=="_u70" keep if experiment_duration<70
		if "`restrict'"=="_o70" keep if experiment_duration>=70

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
		gcollapse diff=did (semean) se=did, by(time)
		gen ub = diff + 1.96*se
		gen lb = diff - 1.96*se
		
		sum time if abs(time)<=15 & se<0.05
		local tmax=round(r(max),5)
		local tmin=round(r(min),5)
		
		gsort time
		twoway 	(scatter diff time) ///
				(rcap ub lb time) if abs(time)<=15 & se<0.05, ///
				xtitle("Years Since Extension") ytitle("∆ Market Value for Extended Properties" "(Relative to Controls)") ///
				legend(off)  ///
				xlabel(`tmin'(5)`tmax') ylabel(-0.1(0.1)0.3) xline(0, lcolor("gs5") lpattern(dash)) yline(0, lcolor("gs5") lpattern(dash))
		graph export "$fig/event_study`tag'`restrict'.png", replace
	}
}



*****************************************************
* Figure A.22: y* by Extension Amount
*****************************************************

use "$clean/experiments.dta", clear
gen xaxis = _n+$year0-1 if _n+$year0-1 <= $year1

gen oth = !k90 & !k700p
gen all = 1

foreach tag in k90 k700p oth all {
	ystar_timeseries did_rsi "`tag'" "_`tag'"
}

twoway 	(line ystar_all xaxis, lpattern(solid) lcolor("black")) ///
		(line ystar_k90 xaxis if xaxis>2003 & xaxis<2008, lpattern(dash) lcolor("$accent1")) ///
		(line ystar_k90 xaxis if xaxis>2008, lpattern(dash) lcolor("$accent1")) ///
		(line ystar_k700p xaxis, lpattern(shortdash) lcolor("$accent2")) ///
		(line ystar_oth xaxis, lpattern(dash_dot) lcolor("$accent3")) ///
		(rarea ub_all lb_all xaxis, color(gray%30) lcolor(%0)) , ///
		legend(order(1 "All" 2 "90 Year Extensions" 4 "700+ Year Extensions" 5 "Other Extensions") ring(0) position(2)) ///
		xtitle("") ytitle("y*") xlabel(2000(5)2023)
graph export "$fig/ystar_timeseries_by_k.png", replace


*****************************************************
* Figure A.24: Robustness
*****************************************************

use "$clean/experiments_incflippers.dta", clear

foreach tag in "" "_bmn" "_flip" "_yearly" {
	ystar_timeseries did_rsi`tag' "" "`tag'"
}

twoway 	(line ystar xaxis, lpattern(solid) lcolor(black)) ///
		(line ystar_bmn xaxis, lpattern(dash) lcolor("$accent1")) ///
		(line ystar_flip xaxis, lpattern(dash_dot) lcolor("$accent2")) ///
		(line ystar_yearly xaxis, lpattern(shortdash) lcolor("$accent3")) ///
		(rarea ub lb xaxis, color(gray%30) lcolor(%0)), ///
		xtitle("") ytitle("Natural Rate") xlabel(2003(5)2023) ///
		legend(order(1 "Baseline" 2 "Bailey-Muth-Nourse" 3 "Including Flippers" 4 "Yearly Index") ring(0) position(1)) 
graph export "$fig/robustness_checks.png", replace


*****************************************************
* Figure A.25: Restrictive Controls
*****************************************************
import delimited "$working/rsi/restrictive_controls.csv", clear case(preserve)

gen d = date(date_trans, "MDY")
drop date_trans
rename d date_trans

gen did_rc = d_log_price - d_log_price_ctrl
foreach var of varlist num_controls radius {
	rename `var' `var'_rc
}
keep property_id date_trans did_rc *log_price_ctrl num_controls radius

* Winsorize
winsor did_rc, p(0.01) gen(did_rc_win)
replace did_rc = . if did_rc!=did_rc_win
drop did_rc_win

tempfile rc
save `rc'

use  "$clean/experiments.dta", clear
merge 1:1 property_id date_trans using `rc', nogen keep(master match)
keep if !missing(did_rsi) & !missing(did_rc)

ystar_timeseries did_rsi "" "_rsi"
ystar_timeseries did_rc "" "_rc"

twoway 	(line ystar* xaxis) ///
		(rarea ub_rsi lb_rsi xaxis, color(gray%30) lcolor(%0)), ///
		legend(order(1 "Repeat Sales" 2 "Exact Match") ring(0) position(1)) ///
		xtitle("") ytitle("y*") xlabel(2003(5)2023)
graph export "$fig/restrictive_controls.png", replace

*****************************************************
* Figure A.26: Fixed Regional Composition
*****************************************************

use "$clean/experiments.dta", clear

qui: sum year
local y = r(max) - r(min)

bys lpa_code_n: gen N_i = _N/`y' // Average observations per year
bys lpa_code_n year: gen N_it = _N

// Composition in 2022 
gen temp=1 if year==2022 
gegen N_2022 = total(temp), by(lpa_code_n)

gen w1 = N_i/N_it
gen w2 = N_2022/N_it

ystar_timeseries did_rsi "" "1"
ystar_timeseries did_rsi "" "2" "w1"
ystar_timeseries did_rsi "" "3" "w2"

twoway  (line ystar1 ystar2 ystar3 xaxis) ///
		(rarea ub1 lb1 xaxis, color(gs10%30) lcolor(%0)), ///
		legend(order(1 "Baseline" 2 "Fixed Composition (Mean)" 3 "Fixed Composition (2022)") ring(0) position(2)) ///
		xlabel(2003(4)2023) xtitle("") ytitle("y*")
graph export "$fig/control_for_composition.png", replace

*****************************************************
* Figure A.27: Instrument First Stage
*****************************************************

use "$clean/ystar_by_lpas_2009-2022.dta", clear
binscatter2 refusal_maj_7908 delchange_maj5 [aw=w], ytitle("Refusal Rate") xtitle("Change in Delay Rate")
graph export "$fig/supply_elasticity_first_stage.png", replace

*****************************************
* Figure A.30: Time on Market
*****************************************

use "$clean/leasehold_flats.dta", clear
reghdfe time_on_market i.bedrooms i.floorarea_50 i.age_50, absorb(i.year##i.quarter##i.outcode) residuals(tom_res)
gen duration_yr = int(duration)
gcollapse time_on_market=tom_res (semean) se=tom_res, by(duration_yr)
gen ub = time_on_market + 1.96*se
gen lb = time_on_market - 1.96*se
twoway  (scatter time_on_market duration) ///
		(rcap ub lb duration) if duration>=40 & duration<=100, ///
		legend(off) xtitle("Duration") ytitle("Time on Market, Residuals")  ///
		xlabel(40(20)100)
graph export "$fig/time_on_market.png", replace

*************************************
* Figure A.31: Liquidity Premium
*************************************

global r_lp = 4
global lambda = 70 
global d = 1
twoway ( function y=ln(  ((1-exp(-($r_lp /100) * (x+90-$lambda ) ))/($r_lp /100)) ///
			+ ((1-exp(-(($r_lp +$d )/100) * $lambda )) * (exp(-($r_lp /100) * (x+90-$lambda ) )) / (($r_lp +$d )/100) ) ) ///
	   - ln(  ((1-exp(-($r_lp /100) * max(0, (x-$lambda ) ) ))/($r_lp /100)) ///
			+ ((1-exp(-(($r_lp +$d )/100) * min(x,$lambda ) )) * (exp(-($r_lp /100) * max(0, (x-$lambda )) ) ) / (($r_lp +$d )/100) ) ), ///
			range(20 150)) ///
		(function y=ln(1-exp(-($r_lp /100) * (x+90) )) - ln(1-exp(-($r_lp /100) * x)), range(20 150) lpattern(dash)), ///
		legend(order(1 "With Liquidity Premium" 2 "Without Liquidity Premium")) ///
		xtitle("Duration") ytitle("Price Difference After Extension vs. Control") ///
		xline($lambda)
graph export "$fig/liqudity_premium_predicted_values.png", replace

*************************************
* Figure A.32: Risk Premium
*************************************
use "$clean/risk_premium.dta", clear
gen date = year + (quarter-1)/4 

twoway (line risk_premium date, lcolor("$accent1")) ///
		(line g_balanced date, lcolor("$accent2")), ///
		legend(order(1 "Long-Run Risk Premium" 2 "Long-Run Capital Gains")) ///
		xtitle("") ytitle("") ///
		xlabel(2000(5)2025) ylabel(0(1)6)
graph export "$fig/var_results.png", replace

*******************************************
* Figure A.33: Refusal Rate Endogeneity
*******************************************

use "$clean/ystar_by_lpas_2009-2022.dta", clear
drop if missing(d_ystar)
gen central_london = pdevel >=0.8	

reg refusal_maj_7908 delchange_maj5
predict refusal_predicted						

twoway 	(scatter d_log_hpi   refusal_maj_7908 if !central_london, msymbol(O) mcolor(gs10)) ///
		(scatter d_log_hpi  refusal_maj_7908 if central_london, mcolor("$accent1") msymbol(D)), ///
		legend(order(1 "Share Developed Less Than 80%" 2 "Share Developed More Than 80%") ring(0) position(5)) ///
		xtitle("Refusal Rate") ytitle("Average Price Growth")
graph export "$fig/refusal_rate_endogeneity.png", replace

twoway 	(scatter d_log_hpi   refusal_predicted if !central_london, msymbol(O) mcolor(gs10)) ///
		(scatter d_log_hpi  refusal_predicted if central_london, mcolor("$accent1") msymbol(D)), ///
		legend(order(1 "Share Developed Less Than 80%" 2 "Share Developed More Than 80%") ring(0) position(5)) ///
		xtitle("Predicted Refusal Rate") ytitle("Average Price Growth")
graph export "$fig/refusal_rate_endogeneity_corrected.png", replace
 
**********************************************
* Figure A.34 + Table A.6: Discontinuity Test
**********************************************

use "$clean/leasehold_flats.dta", clear
replace duration = round(duration)
replace L_duration = round(L_duration)

// Drop extensions and flippers
drop if years_held<2
drop if extension
keep if L_year>=2003

local lb = 60
local ub = 100

gen coeff_post = .
gen coeff_pre = .
gen cutoff = _n + `lb' - 1 if _n + `lb' - 1 <= `ub'

gen d_log_price_ann = d_log_price100/(days_held/365.25)

gen crossed=.
gen crossed80 = duration<=80 & L_duration>82
forv c = `lb'/`ub' {
	di `c'
	qui: if `c' != 80 replace crossed =  duration<=`c' & L_duration>`c'+2 & !crossed80
	qui: else replace crossed = crossed80
	
	qui: sum d_log_price_ann if crossed==1 & L_year<=2010 & duration >= `c'-10 & L_duration <=`c' + 10
	if r(N) > 20 {
		qui: reghdfe d_log_price_ann i.crossed if L_year<=2010 & duration >= `c'-10 & L_duration <=`c' + 10, absorb(year##L_year##lpa_code_n)
		qui: replace coeff_pre = _b[1.crossed] if cutoff==`c'	
	}
	
	qui: sum d_log_price_ann if crossed==1 & L_year>2010 & duration >= `c'-10 & L_duration <=`c' + 10
	if r(N) > 20 {
		qui: reghdfe d_log_price_ann i.crossed if L_year>2010 & duration >= `c'-10 & L_duration <=`c' + 10, absorb(year##L_year##lpa_code_n)
		qui: replace coeff_post = _b[1.crossed] if cutoff==`c'
	}
}

gen l1 = 1 
gen l2 = 2
twoway 	(scatter l1 coeff_pre, mcolor(gray) msymbol(Oh) msize(vlarge)) ///
		(scatter l2 coeff_post, mcolor(gray) msymbol(Oh) msize(vlarge)) ///
		(scatter l1 coeff_pre if cutoff==80, mcolor("$accent1") msymbol(O) msize(vlarge)) ///
		(scatter l2 coeff_post if cutoff==80, mcolor("$accent1") msymbol(O) msize(vlarge)) if cutoff>70, ///
		legend(order(3 "Discontinuity Experiment at 80" 1 "Placebo Experiments Away from 80") rows(2) ring(0) position(2)) ///
		ylabel(0 " " 1 "Pre-2010" 2 "Post 2010" 3 " ") xlabel(-1(0.5)1)  ///
		xtitle("Coefficient")
graph export "$fig/discontinuity_at_80.png", replace

eststo clear
local c = 80
qui eststo: reghdfe d_log_price_ann i.crossed80 if L_year<=2010 & duration >= `c'-10 & L_duration <=`c' + 10, absorb(year##L_year##lpa_code_n) cluster(year L_year lpa_code_n)
qui: estadd local FE "\checkmark", replace
qui: estadd local period "Pre 2010", replace

qui eststo: reghdfe d_log_price_ann i.crossed80 if L_year>2010 & duration >= `c'-10 & L_duration <=`c' + 10, absorb(year##L_year##lpa_code_n) cluster(year L_year lpa_code_n)
qui: estadd local FE "\checkmark", replace
qui: estadd local period "Post 2010", replace

esttab using "$tab/discontinuity_at_80.tex", ///
	nomtitle ///
	keep(1.crossed80) varlabels(1.crossed80 "Crossed Cutoff") ///
	stats(FE period N, label("Sale Year x Purchase Year x LA FE" "Period") fmt(%9.0gc)) ///
	 replace b(2) se(2)

*************************************
* Figure A.35: Hazard Rate
*************************************

use "$clean/leasehold_panel.dta", clear
gen period = 1 if year<2010 
replace period = 2 if year>=2010 & year<2020
collapse (mean) extended, by(L_duration_bin period)
replace extended = extended * 100
keep if L_duration<=150
twoway 	(line extended L_duration if period==1, lpattern(solid) lcolor(grey) lwidth(medthick)) ///
		(line extended L_duration if period==2, lpattern(solid) lcolor("$accent1") lwidth(medthick)) if L_duration>10 & L_duration<=125, ///
		legend(order(1 "Pre 2010" 2 "Post 2010") ring(0) position(2)) ///
		xtitle("Duration") ytitle("% Extended") ///
		xline(80, lcolor(red) lpattern(dash))
graph export "$fig/hazard_rate_2period.png", replace

*************************************
* Figure A.36: Option Value Correction
*************************************

use "$clean/experiments.dta", clear
scalar alpha = 0.53

foreach tag in "" "_corrected" {
	gen ystar`tag' = .
	gen ub`tag' = .
	gen lb`tag' = .	
}
gen xaxis = _n+$year0-1 if _n+$year0-1 <= $year1
gen over80 = T>80

forv year=$year0(1)$year1 {
	// Baseline
	qui: nl $nlfunc if year==`year', initial(ystar 3) variables(did_rsi T k)
	qui: replace ystar = _b[/ystar] if xaxis==`year'
	qui: replace ub = _b[/ystar] + 1.96 * _se[/ystar] if xaxis==`year'
	qui: replace lb = _b[/ystar] - 1.96 * _se[/ystar] if xaxis==`year'
	
	// Corrected
	qui: nl (did_rsi = ln(1-exp(-$ystar_func  *(T+k))) ///
			- ln(1-exp(-$ystar_func * T) + ///
			(over80 * Pi * alpha) * (exp(-$ystar_func * T) - exp(-$ystar_func  * (T+90))) )) if year==`year', initial(ystar 3) variables(did_rsi T k)

	qui: replace ystar_corrected = _b[/ystar] if xaxis==`year'
}

twoway 	(line ystar ystar_corrected xaxis, lcolor(black "$accent1")) ///
		(rarea ub lb xaxis, color(gray%30) lcolor(%0)), ///
		legend(order(1 "Baseline" 2 "Corrected for Option Value") ring(0) position(2)) ///
		xtitle("") ytitle("y*") xlabel($year0(5)$year1)
graph export "$fig/ystar_timeseries_corrected.png", replace
 
 
*************************************
* Figure A.37: Sportelli
*************************************

use "$clean/leasehold_panel.dta", clear
gen period = 1 if year>=2003 & year<=2006
replace period = 2 if year>2006 & year<2010
replace period = 3 if year>=2010
collapse (mean) extended, by(L_duration_bin period)
replace extended = extended * 100
keep if L_duration<=150

twoway 	(line extended L_duration if period==1, lpattern(solid) lcolor("$accent2") lwidth(medthick)) ///
		(line extended L_duration if period==2, lpattern(solid) lcolor(grey) lwidth(medthick)) ///
		(line extended L_duration if period==3, lpattern(solid) lcolor("$accent1") lwidth(medthick)) if L_duration>10 & L_duration<=125, ///
		legend(order(1 "2003-2006" 2 "2006-2010" 3 "2010-2020") ring(0) position(2)) ///
		xtitle("Duration") ytitle("% Extended") ///
		xline(80, lcolor(red) lpattern(dash))
graph export "$fig/hazard_rate_2period_sportelli.png", replace

*************************************
* Table A.1: English Housing Survey
*************************************

use "$clean/ehs.dta", clear
gen lh_str = "Leasehold" if leasehold 
replace lh_str = "Freehold" if !leasehold
eststo clear
estpost tabstat income age has_mortgage ltv, by(lh_str) statistics(mean semean) columns(statistics) nototal
esttab using "$tab/lh_fh_stats.tex", ///
	main(mean %9.2fc ) aux(semean %9.2fc) nostar unstack nonum stats(N, fmt(%9.0gc)) ///
	varlabels(income "Income" age "Age" ltv "LTV" has_mortgage "\% Have Mortgage" ) ///
	nonotes replace
	
*****************************************
* Table A.4: Mortgage Statistics
*****************************************
use "$clean/ehs.dta", clear
eststo clear
estpost tabstat mortgagelength ltv has_mortgage varrate, by(length_at_purchase) statistics(mean semean) columns(statistics)
esttab using "$tab/mortgage_stats.tex", ///
	main(mean "1") aux(semean "1") nostar unstack nonum stats(N, fmt(%9.0gc)) ///
	varlabels(mortgagelength "Mortgage Length" ltv "LTV" has_mortgage "\% Have Mortgage" varrate "\% Adjustable Rate") ///
	nonotes addnotes("mean reported; standard error of mean in parentheses") ///
	replace
	
*************************************
* Table A.5: Seasonality
*************************************
use "$clean/ystar_quarterly.dta", clear
gen w = 1/var

eststo clear
eststo: reghdfe ystar i.quarter [aw=w], absorb(year) vce(robust)
estadd local fe "\checkmark", replace
esttab using "$tab/seasonality.tex", ///
	nomtitle ///
	keep(2.quarter 3.quarter 4.quarter) ///
	varlabels(2.quarter "2nd Quarter" 3.quarter "3rd Quarter" 4.quarter "4th Quarter") ///
	stats(fe N, label("Year FE") fmt(%9.0gc)) ///
	 replace b(2) se(2)

*************************************
* Table A.8: Sportelli Case
*************************************

use "$clean/leasehold_flats.dta", clear
replace duration = round(duration)
replace L_duration = round(L_duration)

drop if years_held<2
drop if extension
keep if L_year>=2003

gen d_log_price_ann = d_log_price100/(days_held/365.25)
gen crossed80 = duration<=80 & L_duration>82

eststo clear
local c = 80
qui eststo: reghdfe d_log_price_ann i.crossed80 if L_year>=2003 & L_year<=2006 & duration >= `c'-10 & L_duration <=`c' + 10, absorb(year##L_year##lpa_code_n) cluster(year L_year lpa_code_n)
qui: estadd local FE "\checkmark", replace
qui: estadd local period "2003-2006", replace

qui eststo: reghdfe d_log_price_ann i.crossed80 if L_year>2006 & L_year<=2010 & duration >= `c'-10 & L_duration <=`c' + 10, absorb(year##L_year##lpa_code_n) cluster(year L_year lpa_code_n)
qui: estadd local FE "\checkmark", replace
qui: estadd local period "2006-2010", replace

qui eststo: reghdfe d_log_price_ann i.crossed80 if L_year>2010 & duration >= `c'-10 & L_duration <=`c' + 10, absorb(year##L_year##lpa_code_n) cluster(year L_year lpa_code_n)
qui: estadd local FE "\checkmark", replace
qui: estadd local period "2010-2023", replace

esttab using "$tab/discontinuity_at_80_sportelli.tex", ///
	nomtitle ///
	keep(1.crossed80) varlabels(1.crossed80 "Crossed Cutoff") ///
	stats(FE period N, label("Sale Year x Purchase Year x LA FE" "Period") fmt(%9.0gc)) ///
	 replace b(2) se(2)

	 
*****************************************
* Extension Amount Histogram
*****************************************

use "$clean/experiments.dta", clear
histogram extension_amount if extension_amount<1000, xtitle("Extension Amount") percent width(5) start(30)
graph export "$fig/extension_amount_histogram.png", replace
