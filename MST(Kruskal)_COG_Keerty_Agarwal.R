#############
#Code to create a Minimal Spanning tree, using the Kruskal's algorithm 
#and also reduce to 3 clusters using center of gravity
#developed by: Keerty Agarwal
#Date: 8th May 2018, United Kingdom
#############

#Packages which would be required
install.packages("ggmap")
install.packages("igraph")
install.packages("geosphere")
install.packages("optrees")
install.packages("ggrepel")
install.packages("SDMTools")

# older versions of pacakages to be loaded from github due to errors with latest ones
devtools::install_github("dkahle/ggmap")
devtools::install_github("hadley/ggplot2")

#load libraries
library(ggmap)
library(igraph)
library(geosphere)
library(optrees)
library(ggrepel)
library(SDMTools)

#create a data frame with longitude and lattitude of all demand nodes
lonlat <- read.table(textConnection(
  "lat, lon
  53.4807593,-2.24263050000001
51.463024,0.360498000000006
  48.856614,2.35222190000001
  40.4167754,-3.70379019999995
  45.4642035,9.18998199999998
  48.1351253,11.5819804999999
  64.1265205999999,-21.8174391999999
  42.6977082,23.3218675
  50.1109221,8.68212670000002
  50.0755381,14.4378004999999
  43.7695604,11.2558136
  41.3850639,2.17340349999994
  "),header=TRUE,strip.white = TRUE, sep=",")

#Vector of names of demand nodes
nname = c( "Manchester",
           "Tilbury",
           "Paris",
           "Madrid",
           "Milan",
           "Munich",
           "Reykjavik",
           "Sofia",
           "Frankfurt",
           "Prague",
           "Florence",
           "Barcelona")

#demand required on each node ( in tonnes)
demand=c(75,
         220,
         250,
         150,
         100,
         125,
         25,
         47,
         111,
         31,
         45,
         80)

#putting it all together in a dataframe
v = data.frame(
  ids  = 1:12,
  name = nname,
  x  = lonlat$lon,
  y  = lonlat$lat)

#Let us determine the distance matrix `D` between all nodes.
D <- distm(lonlat, lonlat, fun=distVincentyEllipsoid) # in meters

#The distance matrix is transformed to an edge "list" (actually a matrix) so that we can build the MST later.
mat2list <- function(D) {
  n = dim(D)[1]
  k <- 1
  e <- matrix(ncol = 3,nrow = n*(n-1)/2)
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      e[k,] = c(i,j,D[i,j])
      k<-k+1
    }
  }
  return(e)
}

#edges in km
eD = mat2list(D/1000)

#applying the Kruskal's algorithm to the given edge list
kMSTree <- msTreeKruskal(1:12, eD)

# Extracting the Arc network from Kruskal MST output
kMSTree_net <- kMSTree[[2]]

kMSTree_net


#To draw two line segments we have to introduce groups, 
#i.e. line segments belonging to one path 
net <- graph.data.frame(kMST_DF[,1:2],directed = FALSE, vertices = v)
R = mst2lines(net, lonlat)

#display the graph
range <- (apply(lonlat,2,max) - apply(lonlat,2,min))*.10
xlimits = c(min(lonlat$lon)-range[1],max(lonlat$lon)+range[1])
ylimits = c(min(lonlat$lat)-range[2],max(lonlat$lat)+range[2])

map <- qmap('Europe',zoom=3, maptype='hybrid')

map + coord_map(xlim = xlimits, ylim = ylimits)+
  geom_path(aes(x = lon, y = lat, group=group), data = R, colour = 'red', size = 3
  )+
  geom_point(data = v, aes(x = x, y = y), color="yellow", size=10, alpha=0.5)+
  geom_label_repel(data = v, aes(x = x, y = y, label=name),size=4,
                   point.padding = unit(0.5, "lines"))

#To find the 3 clusters we need to remove the largest arcs till the time we find 3 clusters
#order by edges
kMSTree_net[order(kMSTree_net[,3]),]

kMSTree_net

#removing 1st largest edge 11(Reykjavik - Manchester), we still have 1 cluster
#Removing the 2nd largest edge 10(Sofia - Prague), we still have only cluster
#removing the 3rd largest edge 9(Paris - Barcelona), we get 2 clusters
#Removing the 4th largest edge 8(Paris - Franfurt), we get 3 clusters 
#Cluster1 with Manchester, Tilbury, Paris
#Cluster2 with Madrid, Barcelona
#Cluster3 with Frankfurt,Munich, Prague, Milan and Florence
#We will now calculate the center of gravity for each of these clusters
# We can visualize the clusters
rrow<-nrow(kMSTree_net) -4
#Remaining arcs
c_kM<-kMSTree_net[1:rrow,]

#Redo the groups and mst2lines and plot the remaning edges to identify the clusters
c_net <- graph.data.frame(c_kM[,1:2],directed = FALSE, vertices = v)
c_R = mst2lines(c_net, lonlat)

map <- qmap('Europe',zoom=3, maptype='hybrid')

map + coord_map(xlim = xlimits, ylim = ylimits)+
  geom_path(aes(x = lon, y = lat, group=group), data = c_R, colour = 'red', size = 3
  )+
  geom_point(data = v, aes(x = x, y = y), color="yellow", size=10, alpha=0.5)+
  geom_label_repel(data = v, aes(x = x, y = y, label=name),size=4,
                   point.padding = unit(0.5, "lines"))

#calculating center of gravity for cluster1
clus1<-v[c(1,2,3),]
dem1<-demand[c(1,2,3)]

clus2<-v[c(4,12),]
dem2<-demand[c(4,12)]

clus3<-v[c(5,6,9,10,11),]
dem3<-demand[c(5,6,9,10,11)]

#calculating gravity
gravity1<-COGravity(clus1[,3],clus1[,4],z=NULL,dem1)
  
#As the central point is in water with a standard deviation, the limits are plotted  
latlimits1<-c(gravity1[3] - gravity1[4],gravity1[3] + gravity1[4])
lonlimits1<-c(gravity1[1] - gravity1[2],gravity1[1] + gravity1[2])
map <- qmap('Europe',zoom=3, maptype='hybrid')
map + coord_map(xlim = lonlimits1, ylim = latlimits1)

#The northeast extreme point in the given range is
coord1<-c(lonlimits1[1],latlimits1[2])
add1<-revgeocode(coord1, output ="more")
city1<-add[1,c(6,9)]

#Similarly for cluster2
gravity2<-COGravity(clus2[,3],clus2[,4],z=NULL,dem2)
coord2<-gravity2[c(1,3)]
add2<-revgeocode(coord2, output ="more")
city2<-add2[1,c(3,7)]

#Similarly for cluster3
gravity3<-COGravity(clus3[,3],clus3[,4],z=NULL,dem3)
coord3<-gravity3[c(1,3)]
add3<-revgeocode(coord3, output ="more")
city3<-add3[1,c(3,7)]

#As the central point is in water with a standard deviation, the limits are plotted  
latlimits2<-c(gravity2[3] - gravity2[4],gravity2[3] + gravity2[4])
lonlimits2<-c(gravity2[1] - gravity2[2],gravity2[1] + gravity2[2])
map <- qmap('Europe',zoom=3, maptype='hybrid')
map + coord_map(xlim = lonlimits1, ylim = latlimits1)

#The northeast extreme point in the given range is
coord1<-c(lonlimits[1],latlimits[2])
add1<-revgeocode(coord, output ="more")
city1<-add[1,c(6,9)]

coord1<-gravity[c(1,3)]
#library sdmtools is used to find the center of gravity function
#calculate the Centre of Gravity for these points
gravity= COGravity(s_v[,3],s_v[,4],z=NULL,demand)
coord<-gravity[c(1,3)]

#retrieve the address, city and country from geocodes
add<-revgeocode(coord, output ="more")
city<-add[1,c(3,6)]

#Final clusters
c_nnames<-c("Cluster1: Leicester, UK","Cluster2: Setiles, Spain","Cluster3: Durach, Germany")
c_lonlat<-read.table(textConnection(
  "lon, lat
  -1.089427, 52.704575
  -1.659549, 40.753571 
  10.39938, 47.68834 
  "),header=TRUE,strip.white = TRUE, sep=",")

dist<-data.frame(
  ids  = 1:3,
  name = c_nnames,
  x  = c_lonlat$lon,
  y  = c_lonlat$lat)

#display the graph
# range <- (apply(c_lonlat,2,max) - apply(c_lonlat,2,min))*.10
# xlimits = c(min(c_lonlat$lon)-range[1],max(c_lonlat$lon)+range[1])
# ylimits = c(min(c_lonlat$lat)-range[2],max(c_lonlat$lat)+range[2])
map <- qmap('Europe',zoom=3, maptype='hybrid')

map + coord_map(xlim = xlimits, ylim = ylimits)+
  geom_point(data = dist, aes(x = x, y = y), color="yellow", size=10, alpha=0.5)+
  geom_point(data = v, aes(x = x, y = y), color="red", size=5, alpha=0.5)+
geom_label_repel(data = dist, aes(x = x, y = y, label=name),size=4,
                 point.padding = unit(0.5, "lines"))+
  geom_label_repel(data = v, aes(x = x, y = y, label=name),size=3,
                   point.padding = unit(0.5, "lines"))
