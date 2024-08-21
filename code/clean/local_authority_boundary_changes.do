import delimited "$raw/ons/Code_History_Database_(December_2021)_UK/Changes.csv", clear
* Keep England
keep if strpos(geogcd, "E")==1
keep if strpos(geogcd_p, "E")==1

forv year=2009/2021 {
	di `year'
	local pyear = `year'-1
	preserve
		qui: keep if year==`year'
		rename geogcd_p gss_code`pyear'
		rename geogcd gss_code`year'
		
		keep gss_code`pyear' gss_code`year'
		qui: gduplicates drop 
		
		if `year'!=2009 {
			qui: joinby gss_code`pyear' using "$working/map_gss_over_time.dta", unmatched(both) _merge(_merge) 
			qui: replace gss_code`year' = gss_code`pyear' if _merge==2
			keep gss_code????
		}
		
		qui: tempfile map_gss_over_time
		qui: save "$working/map_gss_over_time.dta", replace
	restore
}
