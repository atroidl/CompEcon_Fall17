# Install and require various packages for analysis
install.packages("haven")
library(haven)
install.packages("sandwich")
install.packages("foreign")
install.packages("MASS")
install.packages("texreg")
require(foreign)
require(MASS)
require(sandwich)
require(texreg)

# Read Stata data file used for PS #6

ps6_data <- read_dta( "/Users/alexandradinu/Desktop/CompEcon_Fall17/ProblemSets/count22.dta")

#Regression number 1 Poisson

summary(m1 <- glm(investments ~ union + laborforce + uenemprate + hmeansalaryall + poverty + airport + lanemilesrural + lanemilesurban + landarea + highschool + associates + bachelors + graduateprofessional, family="poisson", data=ps6_data))

#Regression number 2 Poisson with region fe's

summary(m2 <- glm(investments ~ union + laborforce + uenemprate + hmeansalaryall + poverty + airport + lanemilesrural + lanemilesurban + landarea + highschool + associates + bachelors + graduateprofessional + Southeast + NewEngland + GreatLakes + Plains + Southwest + RockyMtn + FarWest
                  , family="poisson", data=ps6_data))

#Regression number 3
summary(m3 <- glm.nb(investments ~ union + laborforce + uenemprate + hmeansalaryall + poverty + airport + lanemilesrural + lanemilesurban + landarea + highschool + associates + bachelors + graduateprofessional + Southeast + NewEngland + GreatLakes + Plains + Southwest + RockyMtn + FarWest
                     , data=ps6_data))

# put models into a LaTeX table
print(texreg(list(m1, m2, m3), dcolumn = TRUE, booktabs = TRUE, digits = 5,
             use.packages = FALSE, custom.model.names = c('Poisson', 'Possion FE', 'NegativeBinomial'),
             custom.coef.names = c('Intercept', 'Union', 'LaborForce', 'UnemploymentRate','AvgHourlySalary', 'Poverty', 'Airport', 'RuralLanes', 'UrbanLanes', 'LandArea' ,'HighSchool', 'Associates', 'Bachelors', 'Graduate', 'Southeast', 'NewEngland', 'GreatLakes','Plains', 'Southwest', 'RockyMtn', 'FarWest'),
             label = "tab:3", caption = "Three Count Models",
             float.pos = "H"))
