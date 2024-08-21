**************************************************************************************
* Code for Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments
* By Verónica Bäcker-Peral, Jonathon Hazell, and Atif Mian
**************************************************************************************
foreach tag in "" "_may2023" {
	import delimited "$working/extracted_terms_open`tag'.csv", clear

	rename uniqueidentifier unique_id 
	rename term lease_details
	rename osuprn uprn
	rename associatedpropertydescriptionid id
	keep id lease_details number_years date_from date_registered date_to merge_key uprn

	* Convert to date type
	gen date_to_d = date(date_to, "MDY")
	gen date_from_d = date(date_from, "MDY")
	gen date_registered_d = date(date_registered, "DMY")

	drop date_to date_from date_registered
	rename date*_d date*
	format date*  %tdDD-NN-CCYY

	* If missing lease start date, we can use lease end date to infer duration. 
	replace date_from = date_to - round(number_years*365.25) if missing(date_from) & !missing(date_to) & !missing(number_years)

	replace number_years = datediff(date_registered, date_to, "year") if missing(number_years) & missing(date_from) & !missing(date_to)
	replace date_from = date_registered if missing(date_from) & !missing(date_to) 

	drop if missing(date_from) | missing(number_years) | number_years < 0

	***********************
	* deal with duplicates
	***********************
	* drop in terms of all variables
	gduplicates drop merge_key number_years date_from, force

	* If duplicates refer to a very long lease, only keep one of them 
	gen duration2023 = number_years - datediff(date_from, date("January 1, 2023", "MDY"), "year")
	gegen min_duration2023 = min(duration2023), by(merge_key)
	drop if min_duration2023 > 300 & duration2023!=min_duration2023

	* If duplicates refer to a very similar lease, keep one. If not, drop them.
	gegen mean_duration2023 = mean(duration2023), by(merge_key)
	gegen sd_duration2023 = sd(duration2023), by(merge_key)
	drop if sd_duration2023 > 10 & sd_duration2023 != .
	gduplicates drop merge_key, force

	*******************************
	* Some more cleaning
	*******************************
	// Extract postcode
	gen postcode = regexs(0) if regexm(merge_key, "[A-Z][A-Z]?[0-9][0-9]?[A-Z]? [0-9][A-Z][A-Z]")
	drop if missing(postcode)
	recast str200 merge_key
	compress
	save "$working/lease_data`tag'.dta", replace
}

* for merge:
use "$working/lease_data.dta", clear
gen merge_key_1 = merge_key 
gen merge_key_2 = merge_key 
keep merge_key* uprn postcode
gen address = strtrim(subinstr(merge_key, postcode, "", .))
export delimited "$working/lease_data_for_merge.csv", replace

******************** Clean purchased lease terms ********************
import delimited "$working/extracted_terms_closed.csv", clear
cap drop v1 
gen closed_lease = 1
tempfile closed_leases
save `closed_leases', replace

import delimited "$working/extracted_terms_other.csv", clear
cap drop unnamed* 
gen closed_lease = 0
append using `closed_leases'

rename transaction_id unique_id 
replace unique_id = "{" + unique_id + "}" if closed_lease

foreach var of varlist date_registered deed_date date_from date_to {
	if inlist("`var'", "date_registered", "deed_date") local pattern "DMY"
	else local pattern "MDY"
	gen d = date(`var', "`pattern'")
	drop `var'
	rename d `var'
	format `var' %tdDD-NN-CCYY
}

// If the lease has an expiration date but not an origination date, use the deed date as the origination date 
gen flag = date_from==. & date_to!=.
replace number_years = datediff(deed_date, date_to, "year") if flag & number_years==.
replace date_from = deed_date if flag | (date_from==. & number_years!=.)
drop flag

drop if number_years == . | number_years < 0
drop if date_from == .
duplicates drop unique_id, force

save "$working/extracted_terms_closed.dta", replace

************* Identify extensions since Feb. 2023 *************
use "$working/lease_data_may2023.dta", clear

keep merge_key lease_details number_years date_from date_registered duration2023 id 
rename (merge_key lease_details number_years date_from date_registered duration2023) (merge_key_may2023 lease_details_may2023 number_years_may2023 date_from_may2023 date_registered_may2023 duration2023_may2023)

merge 1:1 id using "$working/lease_data.dta", keepusing(merge_key lease_details number_years date_from date_registered duration2023)

gen extension_amount_may2023 = duration2023 - duration2023_may2023
keep if _merge==3 & extension_amount_may2023>30
keep id number_years_may2023 date_from_may2023 date_registered_may2023
save "$working/extensions_may2023.dta", replace
