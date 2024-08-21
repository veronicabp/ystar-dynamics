**************************************************************************************
* Code for Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments
* By Verónica Bäcker-Peral, Jonathon Hazell, and Atif Mian
**************************************************************************************

**************************************
* Merge transaction + lease data 
**************************************
use  "$working/price_data_leaseholds.dta", clear 
merge 1:1 unique_id using "$working/extracted_terms_closed.dta", keep(match master) 
gen purchased_lease = _merge==3 
replace closed_lease = 0 if missing(closed_lease)
drop _merge

// Mark leases as expired
drop term 
foreach var of varlist date_from date_registered number_years {
	rename `var' `var'_2
}

merge m:1 property_id using "$working/hmlr_merge_keys.dta", keep(match master) nogen
merge m:1 merge_key using "$working/lease_data.dta", keep(match master) nogen

// If we purchased a non-closed lease title for a transaction, use that lease for transactions for which the lease is missing 
gen date_from_p = date_from_2 if purchased_lease & !closed_lease
ereplace date_from = mean(date_from_p) if purchased_lease & number_years==., by(property_id)

gen number_years_p = number_years_2 if purchased_lease & !closed_lease
ereplace number_years = mean(number_years_p) if purchased_lease & number_years==., by(property_id)

// Identify extensions
gen has_been_extended = closed_lease & (year(date_from) + number_years) - (year(date_from_2) + number_years_2) > 30 & date_from != .
gen extension_amount = (date_from-date_from_2)/365.25 + (number_years-number_years_2) if has_been_extended

// If we purchased the title, use that instead of the fuzzy match
replace date_from 		= date_from_2 if purchased_lease
replace number_years 	= number_years_2 if purchased_lease
replace number_years 	= round(number_years)
replace date_registered = date_registered_2 if purchased_lease
replace date_registered = . if closed_lease

rename date_registered_2 date_expired
replace date_expired 	= . if !closed_lease

// Drop if the lease started a year after the transaction or there is no lease
drop if date_from == . | date_from > date_trans + 365

// Merge in post-2023 extensions
merge m:1 id using "$working/extensions_may2023.dta", keep(match master)
gen postmay2023_ext = _merge==3 & !purchased_lease
drop _merge

replace has_been_extended = 1 if postmay2023_ext
replace extension_amount = (date_from-date_from_may2023)/365.25 + (number_years-number_years_may2023) if postmay2023_ext
replace date_from = date_from_may2023 if date_trans < date_registered & postmay2023_ext
replace number_years = number_years_may2023 if date_trans < date_registered & postmay2023_ext

*****************************************
* Append freeholds
*****************************************
append using "$working/price_data_freeholds.dta"

keep property_id date_trans date_from date_registered date_expired deed_date number_years ///
		price type new tenure ppd_category class_title_code lease_details ///
		postcode city district uprn unique_id ///
		multiple_prices multiple_types multiple_tenures ///
		closed_lease purchased_lease has_been_extended extension_amount postmay2023_ext

**************************************
* Generate Useful Variables
**************************************

* Leasehold indicator
gen leasehold = (tenure == "L")
gen freehold = !leasehold

* Calculate information about lease term at each date
* Record lease term at time of transactions
gen years_elapsed = (date_trans - date_from)/365
gen duration = number_years - years_elapsed

* Isolate first component of postcode
gen pos_empty   = strpos(postcode," ")
gen outcode      = substr(postcode,1,pos_empty)
replace outcode  = trim(outcode)
gen incode = substr(postcode, pos_empty, .)
replace incode = trim(incode)
drop pos_empty

* Get area code from postcode 
gen area = substr(postcode, 1, 2)
forvalues i = 0/9 {
	replace area = subinstr(area, "`i'", "", .)
}

* Get sector from postcode 
gen incode1 = substr(incode, 1, 1)
egen sector = concat(outcode incode1), punct(" ")
drop incode1

gen flat = type == "F" 

* Merge in region 
merge m:1 area using "$raw/geography/postcode_area_regions.dta", keep(master match)

* Gen log price 
gen log_price = log(price)
gen log_price100 = log(price)*100

* Make string factor variables numeric
foreach var of varlist outcode sector type new district city area {
	encode `var', gen(`var'_n)
	drop `var'
	rename `var'_n `var'
}
egen postcode_n = group(postcode)

gen year = year(date_trans)
gen quarter = quarter(date_trans)
gen month = month(date_trans)

**************************************
* Drop Incoherent Data
**************************************

* Drop if remaining lease duration is zero or negative
drop if leasehold & duration <= 0

cap drop _merge

save "$working/merged_hmlr.dta", replace

gen address = strtrim(subinstr(property_id, postcode, "", .))
keep if flat
keep uprn property_id postcode address
gduplicates drop 
save "$working/hmlr_for_hedonics_merge.dta", replace

