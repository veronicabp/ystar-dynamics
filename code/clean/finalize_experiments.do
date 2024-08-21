local tags ""  "_linear" "_flip" "_bmn" "_yearly" "_nocons"

foreach tag in "`tags'" {
	qui: import delimited "$working/rsi/rsi`tag'.csv", clear

	qui: gen d = date(date_trans, "MDY")
	drop date_trans
	rename d date_trans
	
	if inlist("`tag'", "_linear", "_all") local var d_pres`tag'
	else local var d_log_price

	qui: gen did_rsi`tag' = `var' - d_rsi
	foreach var of varlist d_rsi num_controls radius {
		rename `var' `var'`tag'
	}
	
	keep property_id date_trans did_rsi`tag' d_rsi num_controls radius

	qui: save "$working/rsi`tag'.dta", replace 
}

use  "$clean/leasehold_flats.dta", clear
keep if extension
cap drop xaxis

foreach tag in "`tags'" {
	merge 1:1 property_id date_trans using "$working/rsi`tag'.dta", nogen keep(master match)
}

// Set would-have-been duration
gen T = whb_duration
gen T_at_ext = L_duration - datediff(L_date_trans, date_extended, "year")
gen T5 = round(T,5)
gen T10 = round(T,10)

gen k = extension_amount
gen k90 = round(extension_amount, 5)==90
gen k700p = extension_amount>700
gen k90u = extension_amount>30 & extension_amount<90 & !k90
gen k200u = extension_amount < 200

gen year2 = int(year/2)*2
gen year5 = int(year/5)*5

* Merge in hazard rate
replace whb_duration = round(whb_duration)
merge m:1 whb_duration using "$clean/hazard_rate.dta", nogen keep(match master)

gen Pi = cum_prob / 100
replace Pi = 0 if Pi == .
drop cum_prob

* Remove 1% of outliers -- there are some cases where transaction prices are unreasonably low (e.g. 9 BS31 1BW,  December 10, 2014) which may refer to lease extesion payments. We have spoken to the Land Registry, and there is no way to verify when transactions refer to lease extension payments. By removing outliers, we exclude these cases where implied price change is too low or too high.
winsor did_rsi, p(0.005) gen(win)
foreach var of varlist did_rsi* {
	replace `var' = . if did_rsi!=win
}

* Keep sample period 
keep if year>=2000

* Drop properties at the very low end of the yield curve
keep if T>20

* Drop properties that were extended within a month of purchase since we don't know if these were extended before or after the transaction
drop if datediff(L_date_trans, date_extended, "month") <= 0

save "$clean/experiments_incflippers.dta", replace

* Drop flippers
drop if years_held<2

save "$clean/experiments.dta", replace

* Save data that can be public

// Drop private data from the Land Registry
drop *number_years *date_from *date_expired *date_registered class_title_code deed_date unique_id

// Drop private data from Rightmove
drop property_id_rm postcode_rm propertytype newbuildflag retirementflag sharedownershipflag auctionflag furnishedflag *parking* currentenergyrating *heating* *condition* list_date* archive_date *date_hmlr hmlrprice *rent* *bedrooms* listingprice *floorarea* *bathrooms* *livingrooms* *yearbuilt* *age* *datelist* listprice* time_on_market price_change_pct *pres* lat_rm lon_rm transtype

// Drop private data from Zoopla
drop *_z* receptions floors listing_status category property_type listingid

save "$clean/experiments_public.dta", replace

