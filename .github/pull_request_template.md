## Title: Summary, imperative, start upper case, don't end with a period
<!-- Ideally, <=50 chars. 50 chars is here..: -->

### Description
<!-- For use in commit message, wrap at 72 chars. 72 chars is here: -->
- *Category*: <!-- one of bugfix, feature, refactor, POC, CI/infrastructure, documentation, 
                   revert, test, release, other/misc -->
- *JIRA issue*: [MIC-XYZ](https://jira.ihme.washington.edu/browse/MIC-XYZ)

<!-- 
Change description – why, what, anything unexplained by the above.
Include guidance to reviewers if changes are complex.
--> 

### Testing
<!--
Details on how code was verified, any unit tests local for the
repo, regression testing, etc. At a minimum, this should include an
integration test for a framework change. Consider: plots, images,
(small) csv file.

*** REMINDER ***
CI WILL NOT RUN ANY TESTS MARKED AS SLOW (CURRENTLY INCLUDES INTEGRATION TESTS).
MANUALLY RUN SLOW TESTS LIKE `pytest --runslow` WITH EACH PR.
-->
- [ ] all tests pass (`pytest --runslow`)
