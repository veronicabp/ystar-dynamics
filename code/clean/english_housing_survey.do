******************************************************************
* Code for: "Measuring the Natural Rate With Natural Experiments"
* Backer-Peral, Hazell, Mian (2023)
* 
* This program merges and cleans data from the English Housing
* Survey microfiles, 1994-2019.
******************************************************************


*********************************************
* Merge data from the English Housing Survey
*********************************************

local folders: dir "$raw/ehs" dirs "*"
foreach folder of local folders {
	local version = substr("`folder'", strpos("`folder'", "stata"), .)
	
	local num = real(substr("`folder'", 6, 4))
	
	if `num' == 6376 {
		continue
	}
	
	if "`version'" == "stata6" | "`version'" == "stata8" {
		local files: dir "$raw/ehs/`folder'/`version'" files "hh*"
		local file: word 1 of `files'
		local year = subinstr("`file'", "hhold", "", .)
		local year = subinstr("`year'", "hhd", "", .)
		local year = subinstr("`year'", "ess", "", .)
		local year = subinstr("`year'", ".dta", "", .)
		
		if real("`year'")<30 {
			local year = 2000 + real("`year'")
		}
		else {
			local year = 1900 + real("`year'")
		}
		
		use "$raw/ehs/`folder'/`version'/`file'", clear
	}
	if "`version'" == "stata9" | "`version'" == "stata11" | "`version'" == "stata"{
		if "`version'" == "stata" {
			if `num' < 8719 local subfolder "stata/stata11"
			else local subfolder "stata/stata13"
		}
		else {
			local subfolder "`version'"
		}
		
		// Get location of relevant files
		local files: dir "$raw/ehs/`folder'/`subfolder'/derived/" files "general*"
		local file: word 1 of `files'
		local files: dir "$raw/ehs/`folder'/`subfolder'/derived/" files "interview*"
		local ifile: word 1 of `files'
		
		if inlist(`num', 8067, 8254, 8386, 8719, 8921) local flag "_sl_protect"
		else if `num' == 8545 local flag "_sl_protected"
		else local flag ""
		
		// Identify year of file
		local year = subinstr("`file'", "generalfs", "", .)
		local year = subinstr("`year'", "`flag'", "", .)
		local year = subinstr("`year'", ".dta", "", .)
		if `num' == 8921 local year = 19
		local year = 2000 + real("`year'")
		
		di "Num: `num'"
		di "Year: `year'"
		di "====================="
		
		// The merge key changes at one point 
		if `num' <= 8067 local mergekey aacode
		else local mergekey serialanon
		
		use "$raw/ehs/`folder'/`subfolder'/derived/`file'", clear
		qui: merge 1:1 `mergekey' using "$raw/ehs/`folder'/`subfolder'/derived/`ifile'", nogen
		qui: merge 1:1 `mergekey' using "$raw/ehs/`folder'/`subfolder'/interview/owner`flag'.dta", nogen
		qui: merge 1:1 `mergekey' using "$raw/ehs/`folder'/`subfolder'/interview/income`flag'.dta", nogen
		qui: merge 1:1 `mergekey' using "$raw/ehs/`folder'/`subfolder'/interview/dwelling`flag'.dta", nogen
		qui: merge 1:1 `mergekey' using "$raw/ehs/`folder'/`subfolder'/interview/hhldtype`flag'.dta", nogen
		gen year = `year'
	}
	
	cap drop in?
	cap drop gorehs
	rename *, lower
	qui: save "$working/ehs`year'.dta", replace	
}


use "$working/ehs1993.dta", clear
forv y=1994/2019 {
	di "`y'"
	append using "$working/ehs`y'.dta"
}

replace year = year+1900 if year < 100

*************************************************
* Identify leaseholds of various durations
*************************************************

* Lease length
gen remaining_lease = 45 if 		(year >  1997 & inrange(lgthln,1,4)) | ///
									(lgthlnnew==50) // 50 years or less
replace remaining_lease = 55 if 	(year >  1997 & lgthln==5) | ///
									(inrange(lgthlnnew,51,60)) // 50-60
replace remaining_lease = 65 if 	(year >  1997 & lgthln==6) | ///
									(inrange(lgthlnnew,61,70)) // 60-70
replace remaining_lease = 75 if 	(year >  1997 & lgthln==7) | ///
									(inrange(lgthlnnew,71,80)) // 70-80								
replace remaining_lease = 85 if 	(year >  1997 & lgthln==8) | ///
									(inrange(lgthlnnew,81,99)) // 80-99
replace remaining_lease = 100 if 	(year >  1997 & lgthln>=9 & lgthln < .) | ///
									(lgthlnnew >= 100 & lgthlnnew < .) // 100+
								
* Lease length at purchase 
gen lenown10 = int(lenown/10)*10 if lenown > 0
replace lenown10 = 0 if inrange(hlong, 1, 6)
replace lenown10 = 10 if hlong==7
replace lenown10 = 20 if hlong==8
replace lenown10 = 30 if hlong==9
replace lenown10 = 40 if hlong==10

gen length_at_purchase = remaining_lease + lenown10
replace length_at_purchase = 100 if length_at_purchase>85 & length_at_purchase<.

* Label
label define duration 45 "Less Than 50 Years" 55 "50-60 Years" 65 "60-70 Years" 75 "70-80 Years" 85 "80-99 Years" 100 "100+ Years"
label values remaining_lease duration
label values length_at_purchase duration
	
* Freehold or leasehold
gen freehold = freeleas==1 | lease==1 | lease2==1
gen leasehold = freeleas==2 | lease==2 | lease2==2 | remaining_lease!=.
	
*************************************************
* Get mortgage + property price information
*************************************************
gen has_mortgage = ten1 == 2 if inrange(ten1,1,2)
replace has_mortgage = has_mortgage*100


forv i=1/10 {
	di "`i'"
	di "=========="
	tab morgperl if morgperl==`i'
	tab morgper2 if morgper2==`i'
}

* Mortgage data 
gen morgper = morgperl if morgperl>=1 & morgperl<=52
replace morgper = morgper2 if morgper2>=1 & morgper2<=52 & missing(morgper)

gen morgpayments = 12 * morgpayu if inlist(morgper,4,5)
replace morgpayments = 52 * morgpayu if morgper==1
replace morgpayments = 26 * morgpayu if morgper==2
replace morgpayments = 17 * morgpayu if morgper==3
replace morgpayments = 6 * morgpayu if morgper==7
replace morgpayments = 8 * morgpayu if morgper==8
replace morgpayments = 9 * morgpayu if morgper==9
replace morgpayments = 10 * morgpayu if morgper==10
replace morgpayments = 4 * morgpayu if morgper==13
replace morgpayments = 2 * morgpayu if morgper==26
replace morgpayments = 1 * morgpayu if morgper==52
replace morgpayments = . if morgpayments < 0

gen morginitial = onorgmrg if onorgmrg>0
gen propprice = onpurpc if onpurpc>0

// Loan to value ratio
gen ltv = 100 * morginitial/propprice 
replace ltv = . if ltv > 100

// Interest rate
gen rate = 100 * morgpayments/morginitial
replace rate = . if rate > 20

// Interest rate type
gen ratetype = 1 if inttype==2 | inttype2==2 | inttype3==2 | inttype4==4
replace ratetype = 2 if inttype==5 | inttype2==5 | inttype3==5 | inttype3==6 | inttype4==1 | inttype4==2
replace ratetype = 3 if inttype==4 | inttype2==4 | inttype3==4 | inttype4==7 
replace ratetype = 4 if missing(ratetype) & (!missing(inttype) | !missing(inttype2) | !missing(inttype3) | !missing(inttype4)) & inttype>0 & inttype2>0 & inttype3>0 & inttype4>0

gen varrate = 100 * (ratetype==1) if !missing(ratetype)
gen fixed5rate = 100 * (ratetype==2) if !missing(ratetype)
gen trackerrate = 100 * (ratetype==3) if !missing(ratetype)

// Mortgage length
gen mortgagelength = morglnth if morglnth>0


*************************************************
* Other property characteristics
*************************************************

* Tenure
gen owner = inlist(ten1, 1, 2)
gen renter = inlist(ten1, 3)

* Type 
gen house = accomhh1 > 0 & accomhh1 <= 3
gen flat = accomhh1 == 4 | accomhh1==5

* Age
gen age = agehrp
replace age = agehrpx if missing(age)
replace age = agehoh if missing(age)
replace age = 20 if agehrp6x == 1 & missing(age)
replace age = 30 if agehrp6x == 2 & missing(age)
replace age = 40 if agehrp6x == 3 & missing(age)
replace age = 50 if agehrp6x == 4 & missing(age)
replace age = 60 if agehrp6x == 5 & missing(age)
replace age = 70 if agehrp6x == 6 & missing(age)
replace age = . if age < 0

* Income
gen income = hhincx
replace income = weekjnt * 52 if weekjnt > 0 & missing(income)

// Drop properties that don't make sense
keep if owner | renter
keep if freehold | leasehold
drop if freehold & leasehold

save "$clean/ehs.dta", replace 
