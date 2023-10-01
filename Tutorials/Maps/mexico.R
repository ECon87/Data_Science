library(maps)
library(ggplot2)

map_data_mx  <-  map_data("world")[map_data("world")$region == "Mexico", ]
map_data_mx  <-  map_data("world", interior = TRUE)[map_data("world")$region == "Mexico", ]



# Valeo - located in Aguascalientes
# Valeo - located in Chihuahua
# Valeo - located in Queretaro
# Valeo - located in State of Mexico
# Valeo - located in San Luis Potosi
# Valeo - located in Tamaulipas
# Luk - located in Mexico City
# Luk - located in Puebla
# ZF Sachs - located in Jalisco
# ZF Sachs - located in Coahuila
# Daimler:
# Santiago Tianguistenco, Mexico, Truck Manufacturing Plant
# Saltillo Truck Manufacturing Plant
# Detroit Diesel Remanufacturing Centers in Mexico
# San Luis Potosi, Mexico Parts Distribution Center

facilities  <- data.frame(
  Owner = c('Valeo',
            'Valeo',
            'Valeo',
            'Valeo',
            'Valeo',
            'Valeo',
            'Luk',
            'Luk',
            'ZF Sachs',
            'ZF Sachs',
            'Daimler',
            'Daimler',
            'Daimler',
            'Eaton'),
  city = c('Aguascalientes',
           'Chihuahua',
           'Queretaro',
           'State of Mexico',
           'San Luis Potosi',
           'Tamaulipas',
           'Mexico City',
           'Puelba',
           'Jalisco',
           'Coahuila',
           'Santiago Tianguistenco',
           'Saltillo',
           'San Luis Polosi',
           'San Luis Polosi'),
  long = c(-102.2916,
           -106.069099,
           -100.3899,
           -99.7233,
           -100.9855,
           -98.8363,
           -99.1332,
           -98.2063,
           -103.3494,
           -101.7068,
           -99.4676,
           -100.9737,
           -100.9855,
           -100.9855),
  lat = c(21.8853,
          28.632996,
          20.5888,
          19.4969,
          22.1565,
          24.2669,
          19.4326,
          19.0414,
          20.6595,
          27.0587,
          19.1804,
          25.4383,
          22.4,
          22.6))

inc_lat <- 0.3
inc_long <- 0.4
cities  <- data.frame(
  city = c('Aguascalientes',
           'Chihuahua',
           'Queretaro',
           'State of Mexico',
           'San Luis Potosi',
           'Tamaulipas',
           'Mexico City',
           'Puelba',
           'Jalisco',
           'Coahuila',
           'Santiago Tianguistenco',
           'Saltillo'),
  long = c(# Aguascalientes
           -102.2916 - inc_long,
           # Chihuahua
           -106.069099,
           # Queretaro
           -100.3899,
           # State of Mexico
           -99.7233 - inc_long,
           # SLP
           # -100.9855,
           -100 + 1.5 * inc_long,
           # Tamaulipas
           -98.8363,
           # Mexico city
           -99.1332 + 1.3,
           # Puelba
           -98.2063 + 2.5 * inc_long,
           # Jalisco
           -103.3494,
           # Coahuila
           -101.7068,
           # Santiago
           -99.4676 - 1.5 * inc_long,
           # Saltillo
          -100.9737),
  lat = c(# Aguascalientes
          21.8853 + inc_lat,
          # Chihuahua
          28.632996 + inc_lat,
          # Queretaro
          20.5888 + inc_lat,
          # State of Mexico
          19.4969 + inc_lat,
          # SLP
          # 22.1565,
          22.4,
          # Tamaulipas
          24.2669 + inc_lat,
          # Mexico city
          19.4326,
          # Puelba
          19.0414,
          # Jalisco
          20.6595 + inc_lat,
          # Coahuila
          27.0587 + inc_lat,
          # Santiago
          19.1804 - inc_lat,
          # Saltillo
          25.4383 + inc_lat))

ggplot() +
  # First layer
  geom_polygon(data = map_data("world"),
               aes(x = long, y = lat, group = group), fill = 'lightgrey') +
  # Second layer: Mexico
  geom_polygon(data = map_data_mx, aes(x = long, y = lat, group = group),
               color = "red", fill = "lightgoldenrodyellow") +
  geom_point(data = facilities, aes(x = long, y = lat, shape = Owner), size = 3) +
  geom_text(data = cities, aes(x = long, y = lat, label = city), size = 5) +
  # scale_shape_manual(values = c(15, 19)) +
  coord_map() +
  # coord_fixed(1, xlim = c(-116, -83), ylim = c(3, 32))
  coord_sf(xlim = c(-116, -87), ylim = c(15, 32)) +
  theme(panel.background = element_rect(fill = 'lightblue'),
        legend.position = c(0.8, 0.8),
        legend.key.size = unit(1, 'cm'),
        legend.title = element_text(size = 20),
        legend.text = element_text(size = 20))

ggsave("/tmp/mexico.png")




## make a df with only the country to overlap
map_data_es <- map_data('world')[map_data('world')$region == "Spain",]
facilities  <- data.frame(
    long = c(-5),
    lat = c(39))
## The map (maps + ggplot2 )
ggplot() +
    ## First layer: worldwide map
    geom_polygon(data = map_data("world"),
                 aes(x=long, y=lat, group = group),
                 color = '#9c9c9c', fill = '#f3f3f3') +
    ## Second layer: Country map
    geom_polygon(data = map_data_es,
                 aes(x=long, y=lat, group = group),
                 color = 'red', fill = 'pink') +
    geom_point(data = facilities, aes(x = long, y = lat, label = 1)) +
    coord_map() +
    coord_fixed(1.3,
                xlim = c(-13.5, 8.5),
                ylim = c(34, 45)) +
    ggtitle("A map of Spain") +
    theme(panel.background =element_rect(fill = 'blue'))
