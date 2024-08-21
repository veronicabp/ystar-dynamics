**************************************************************************************
* Code for Dynamics of the Long Term Housing Yield: Evidence from Natural Experiments
* By Verónica Bäcker-Peral, Jonathon Hazell, and Atif Mian
**************************************************************************************

cap mkdir "$working/rightmove"

// Merge all files

foreach subfolder in "" "UK 2017 - 2022 Rental" "UK 2017 - 2022 Sale" "UK 2012-2016 Sale and Rental" {
	di "`subfolder'"
	local files: dir "$raw/rightmove/`subfolder'" files "*.csv"
	foreach file of local files {
		di "`file'"
		
		if "`file'" == "area_upload.csv"  continue 
		local dtafile = subinstr("`file'",".csv",".dta",.)
		local output_file "`subfolder' `dtafile'"
		local output_file = strtrim("`output_file'")
		
		// Check if file already exists
		cap confirm file "$working/rightmove/`output_file'"
		if _rc == 0 continue
		
		qui: import delimited "$raw/rightmove/`subfolder'/`file'", clear bindquote(strict) maxquotedrows(10000)		
		if c(k)!=0 & _N!=0{
		// Make format consisting
		qui: gen a_=.
		foreach var of varlist *_* {
			local new_name = subinstr("`var'", "_", "", .)
			rename `var' `new_name' 
		}
		
		// For pre-2012 properties, we need to reformat the date_hmlr
		qui: replace firstlistingdate = substr(firstlistingdate, 1, 10)
		
		// For the SW area, the format is a bit different
		if "`subfolder'"=="UK 2017 - 2022 Sale" & "`file'"=="SW Results Sale.csv" {
			foreach var of varlist *date* {
				qui: gen temp= date(`var', "MDY")
				drop `var'
				qui: generate `var' = string(temp, "%tdCCYY-NN-DD")
				drop temp
			}
		}
		
		*If there are multiple entries per listing, for each listing, we just want the first and last listing dates and their prices
		capture confirm variable changedate
		if !_rc {
			// Convert dates to numeric
			qui: gen datelist = date(changedate, "YMD")
			
			gsort listingid datelist
			qui: gegen datelist0 = first(datelist), by(listingid)
			gsort listingid -datelist
			qui: gegen datelist1 = first(datelist), by(listingid)
			
			qui: gen listprice0 = listingprice if datelist==datelist0
			qui: gen listprice1 = listingprice if datelist==datelist1
			qui: gegen listprice0=mean(listprice0), by(listingid) replace
			qui: gegen listprice1=mean(listprice1), by(listingid) replace
			
			gsort -datelist
			cap gduplicates drop listingid, force
			drop changedate
		}
		else {
			qui: gen datelist0 = ""
			qui: gen datelist1 = ""
			qui: gen listprice0 = ""
			qui: gen listprice1 = ""
		}
		
		qui: ds uprn, not
		qui: tostring `r(varlist)', replace force
		qui: destring uprn, replace force

		qui: save "$working/rightmove/`output_file'", replace
		}
	}
}

* Combine everything
clear 
local count=0
local files: dir "$working/rightmove" files "*.dta"
foreach file of local files {
	di "`count'"
	qui: append using "$working/rightmove/`file'"
	local count=`count'+1
}

egen property_id = concat(address1 postcode), punct(" ")
replace property_id = subinstr(property_id, ".", " ", .)
replace property_id = subinstr(property_id, ",", " ", .)
replace property_id = subinstr(property_id, "'", " ", .)
replace property_id = subinstr(property_id, "#", " ", .)
replace property_id = subinstr(property_id, `"""', " ", .)
replace property_id = subinstr(property_id," - ","-", .)
replace property_id = upper(strtrim(stritrim(property_id)))

drop if missing(postcode)
drop if missing(address1)

gen list_date_1st = date(firstlistingdate, "YMD")
gen archive_date = date(archivedate, "YMD")
gen date_hmlr = date(hmlrdate, "YMD")
drop firstlistingdate archivedate hmlrdate

destring datelist datelist0 datelist1, force replace
format *date* %td

destring listingid chimneyid bedrooms listingprice floorarea bathrooms livingrooms yearbuilt hmlrprice listprice0 listprice1 letrentfrequency, force replace

gen date_rm = datelist1
replace date_rm = list_date_1st if missing(date_rm)
drop if missing(date_rm)

rename postcode postcode_rm
rename property_id property_id_rm

// Annualize rental prices 
gen annualized_listingprice = listingprice * letrentfrequency if transtype=="Rent" & !missing(letrentfrequency)
// If missing frequency, assume it is monthly
replace annualized_listingprice = listingprice * 12 if transtype=="Rent" & missing(letrentfrequency)
replace listingprice = annualized_listingprice if transtype=="Rent"

// Save descriptions for flats
preserve 
	keep if propertytype=="Flat / Apartment" | propertytype=="Flat"
	keep property_id_rm date_rm summary 
	gduplicates drop
	save "$working/rightmove_descriptions.dta", replace
restore

// Keep only relevant variables
keep transtype listingid latitude longitude propertytype listingprice bedrooms newbuildflag retirementflag sharedownershipflag auctionflag furnishedflag postcode floorarea bathrooms livingrooms yearbuilt parking currentenergyrating hmlrprice heatingtype condition list_date_1st date_hmlr property_id archive_date address* uprn datelist? listprice? date_rm status

// Drop if address information is missing
drop if postcode==property_id
gduplicates drop property_id date_rm, force

****************************************************************************
* Remove huge (and probably erroneous) hedonics values 
****************************************************************************
replace bedrooms = . if bedrooms > 10 | bedrooms==0
replace bathrooms = . if bathrooms > 5 | bathrooms==0
replace livingrooms = . if livingrooms > 5 | livingrooms==0
replace floorarea = . if floorarea > 500 | floorarea==0

// Encode string variables 
foreach var of varlist parking currentenergyrating heatingtype condition {
	encode `var', gen(`var'_)
	drop `var'
	rename `var'_ `var'
}

// Drop if missing all hedonics
drop if missing(bedrooms) & missing(bathrooms) & missing(livingrooms) & missing(floorarea) & missing(yearbuilt) & missing(condition) & missing(heatingtype) & missing(parking) & missing(currentenergyrating) & transtype=="Sale"

preserve
	keep if transtype=="Sale"
	save "$working/rightmove_sales.dta", replace
	
	keep if propertytype=="Flat / Apartment" | propertytype=="Flat"
	save "$working/rightmove_sales_flats.dta", replace
restore


preserve
	keep if transtype!="Sale"
	save "$working/rightmove_rents.dta", replace
	
	keep if propertytype=="Flat / Apartment" | propertytype=="Flat"
	save "$working/rightmove_rents_flats.dta", replace
restore

// Save merge keys
preserve
	keep property_id_rm propertytype postcode uprn address1
	gduplicates drop
	rename postcode_rm postcode
	rename property_id_rm property_id
	save "$working/rightmove_for_merge.dta", replace
	
	keep if propertytype=="Flat / Apartment" | propertytype=="Flat"
	gduplicates drop
	save "$working/rightmove_for_merge_flats.dta", replace
restore
