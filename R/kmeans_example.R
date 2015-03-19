
###################################################################################
###################################################################################
# an R script to determine the optimal number of k-means clusters to minimize
# the total within group sum of squares with the minimum number of clusters
#
# Author: Sherry Towers
#         smtowers@asu.edu
# created: June 10, 2013
# This script may be freely shared with this header information intact, but
# is not guaranteed to be free of bugs and errors.
###################################################################################

set.seed(12345)
###################################################################################
# read in the data.  This data was obtained from the Global Terrorism Database
###################################################################################
mydata = read.table("afghanistan_terror_attacks_2009_to_2011.txt",header=T)

###################################################################################
# *always* normalize the variables that you want to cluster
# we'll put the normalized variables into a new data frame
###################################################################################
mydata$longb = (mydata$long-mean(mydata$long))/sd(mydata$long)
mydata$latb = (mydata$lat-mean(mydata$lat))/sd(mydata$lat)
mydata_for_clustering=data.frame(long=mydata$longb,lat=mydata$latb)

###################################################################################
# the data frame mydata_for_clustering is ncol=2 dimensional
# and we wish to determine the minimum number of clusters that leads
# to a more-or-less minimum total within group sum of squares
###################################################################################
kmax = 20 # the maximum number of clusters we will examine; you can change this 
totwss = rep(0,kmax)
kmfit = list() # create and empty list
for (i in 1:kmax){
   totwss[i] = kmeans(mydata_for_clustering,centers=i)$tot.withinss
   kmfit[[i]] = kmeans(mydata_for_clustering,centers=i)
}

###################################################################################
# calculate the adjusted R-squared
# and plot it
###################################################################################
n = nrow(mydata_for_clustering)
rsq = 1-(totwss*(n-1))/(totwss[1]*(n-seq(1,kmax)))
mult.fig(1,main="Afghanistan: terrorist incidents 2009-2011")
plot(seq(1,kmax),rsq,xlab="Number of clusters",ylab="Adjusted R-squared",pch=20,cex=2)

###################################################################################
# try to find the elbow point
###################################################################################
v = diff(rsq)
nv = length(v)
fom = v[1:(nv-1)]/v[2:nv]
nclus = which.max(fom)+1
points(nclus,rsq[nclus],col=2,pch=20,cex=2)

###################################################################################
# http://faculty.chicagobooth.edu/matt.taddy/teaching/scripts/IC.R
# provides a function for calculating the AIC statistic 
# see Eqn 197 of 
# http://nlp.stanford.edu/IR-book/html/htmledition/cluster-cardinality-in-k-means-1.html
# k is the number of clusters
# m is the dimensionality of the space
###################################################################################
kmeansAIC = function(fit){
        m = ncol(fit$centers)
        k = nrow(fit$centers)
	D = fit$tot.withinss
	return(D + 2*m*k)
}

aic=sapply(kmfit,kmeansAIC)
mult.fig(1,main="Afghanistan: terrorist incidents 2009-2011")
plot(seq(1,kmax),aic,xlab="Number of clusters",ylab="AIC",pch=20,cex=2)

###################################################################################
# try to find the elbow point
###################################################################################
v = -diff(aic)
nv = length(v)
fom = v[1:(nv-1)]/v[2:nv]
nclus = which.max(fom)+1
points(nclus,aic[nclus],col=2,pch=20,cex=2)

###################################################################################
# for the number of clusters at the "elbow" point, get the k-means object
###################################################################################
myclus = kmeans(mydata_for_clustering,centers=nclus)
print(names(myclus))

#########################################################################
# Now let's overlay our clustered data on a map of Afghanistan
# R shape files for country border polygons from www.gadm.org/country
# First load in the shapefile data for the borders 
#########################################################################
require("sp")
require("rgdal")
con = url("http://gadm.org/data/rda/AFG_adm1.RData")
print(load(con))
close(con)

#########################################################################
# now set up the map projection 
# WGS84 is the coordinate system used by GPS
# http://en.wikipedia.org/wiki/World_Geodetic_System
#########################################################################
projection=CRS("+proj=longlat +datum=WGS84")

#########################################################################
# now transform the polygons for the borders into the map projection
#########################################################################
gadm.projection = spTransform(gadm, projection)

vcen_long = myclus$centers[,1]
vcen_lat = myclus$centers[,2]
vcen_long = (vcen_long*sd(mydata$long))+mean(mydata$long)
vcen_lat = (vcen_lat*sd(mydata$lat))+mean(mydata$lat)
#########################################################################
# now plot the map, and overlay the latitude and longitude of the 
# terrorist attacks, colored by cluster assignment
#########################################################################
mult.fig(1,main="Afghanistan: terrorist incidents 2009-2011")
plot(gadm.projection)
scol = rainbow(nclus,end=0.8) # select colors from the rainbow
points(mydata$long,mydata$lat,col=scol[myclus$cluster],pch=20)
points(vcen_long,vcen_lat,pch="*",cex=6)
points(vcen_long,vcen_lat,col=scol[seq(1,nclus)],pch="*",cex=5.5)
points(mydata$long,mydata$lat,col=scol[myclus$cluster],pch=20,cex=0.25)




