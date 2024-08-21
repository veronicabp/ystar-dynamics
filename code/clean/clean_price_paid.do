**************************************************************************************
* Code for Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments
* By Verónica Bäcker-Peral, Jonathon Hazell, and Atif Mian
**************************************************************************************
import delimited "$raw/hmlr/pp-complete.csv", clear

rename v1  unique_id
rename v2  price
rename v3  date
rename v4  postcode
rename v5  type
rename v6  new
rename v7  tenure
rename v8  street_number
rename v9  flat_number
rename v10 street
rename v11 locality
rename v12 city
rename v13 district
rename v14 county
rename v15 ppd_category

// Clean date
replace date = substr(date, 1, 10)
gen date_trans = date(date,"YMD")
format date_trans %tdDD-NN-CCYY

// Drop missing values
drop if missing(street_number) & missing(flat_number)
drop if missing(postcode)

// Generate unique property id
egen property_id = concat(flat_number street_number postcode), punct(" ")
replace property_id = subinstr(property_id,".","",.)
replace property_id = subinstr(property_id,",","",.)
replace property_id = subinstr(property_id,"'","",.)
replace property_id = subinstr(property_id," - ","-",.)
replace property_id = subinstr(property_id,"(","",.)
replace property_id = subinstr(property_id,")","",.)
replace property_id = subinstr(property_id,"FLAT ","",.)
replace property_id = subinstr(property_id,"APARTMENT ","",.)
replace property_id = upper(strtrim(stritrim(property_id)))

// Drop if unknown property tenure
drop if tenure == "U"

// Drop if unknown property type 
drop if type == "O"

compress
***************************************
* Investigate and drop duplicates
***************************************

// There should only be one entry for each property/date
gsort property_id date_trans -locality -street -city // Prioritize data that is not missing locality
gduplicates tag property_id date_trans, gen(dup)

// If there are two transactions on the same date for PPD category A and B, keep only the one for category A
gduplicates tag property_id date_trans ppd_category, gen(dup_cat)
drop if dup!=dup_cat & ppd_category=="B" 

// If there are multiple types per property/transaction, drop
drop dup
gduplicates tag property_id date_trans, gen(dup)
gduplicates tag property_id date_trans type, gen(dup_type)
drop if dup_type!=dup

// If there are multiple tenures per property/transaction, drop
drop dup 
gduplicates tag property_id date_trans, gen(dup)
gduplicates tag property_id date_trans tenure, gen(dup_tenure)
drop if dup!=dup_tenure

// For the duplicates that remain, just take mean price across them and flag
gegen price_mn = mean(price), by(property_id date_trans)
gen multiple_prices = price!=price_mn
replace price = price_mn if price!=price_mn

gduplicates drop property_id date_trans, force
drop dup*

// Flag properties that are listed as multiple types over the course of their lives 
gduplicates tag property_id, gen(dup_pid)
gduplicates tag property_id type, gen(dup_pid_type)
gen multiple_types = dup_pid != dup_pid_type

// Flag properties that are listed as multiple tenures over the course of their lives 
gduplicates tag property_id tenure, gen(dup_pid_tenure)
gen multiple_tenures = dup_pid != dup_pid_tenure

drop dup*

* Save freeholds
preserve
	keep if tenure == "F"
	save "$working/price_data_freeholds.dta", replace
restore
	
* Save leaseholds
drop if tenure == "F"
save "$working/price_data_leaseholds.dta", replace

// Create merge keys
egen merge_key_1 = concat(flat_number street_number street city postcode), punct(" ")
egen merge_key_2 = concat(flat_number street_number street locality city postcode), punct(" ")

foreach var of varlist merge_key* {
	replace `var' = subinstr(`var',".","",.)
	replace `var' = subinstr(`var',",","",.)
	replace `var' = subinstr(`var',"'","",.)
	replace `var' = subinstr(`var',"(","",.)
	replace `var' = subinstr(`var',")","",.)
	replace `var' = subinstr(`var',"FLAT","",.)
	replace `var' = subinstr(`var',"APARTMENT","",.)
	replace `var' = upper(strtrim(stritrim(`var')))
}

replace merge_key_2 = "" if locality==city | merge_key_1==merge_key_2

// Keep only one entry per property_id
gsort property_id -street -locality -city // Prioritize data that is not missing
gduplicates drop property_id, force

gen address = strtrim(subinstr(property_id, postcode, "", .))
keep property_id postcode address merge_key*

export delimited "$working/price_data_for_merge.csv", replace
