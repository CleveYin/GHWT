library(tidyverse)
library(ggpubr)
library(readxl)
library(ggpmisc)
library(cartography)
library(rgdal)

# figure5: mean yearly historical heat waves
hw_data_1 = read_csv("I:\\1_papers\\6_heat wave variation\\2_data\\7_statistic\\1_historical_summary\\historical_hwe_yearly_1971-2020.csv")
hw_data_1$Attribute <- factor(hw_data_1$Attribute, levels = c("HWF", "HWD", "HWAD", "HWAT", "HWSD", "HWED"))
my.formula <- y ~ x
ft_labels_1 <- c("HWF" = "HWF (times)", "HWD" = "HWD (days)", "HWAD" = "HWAD (days)", "HWAT" = "HWAT (°C)", "HWSD" = "HWSD (DOY)", "HWED" = "HWED (DOY)")
ggscatter(hw_data_1, x = "Year", y = "Value", color = "Data", palette = "jco", add = "reg.line", conf.int = TRUE) +
  stat_poly_eq(formula = my.formula, aes(label = paste(..eq.label.., ..p.value.label.., sep = "~~~"), color = Data), label.x.npc = 0.05, p.digits = 2, vstep = 0.1) +
  facet_wrap(~Attribute, ncol = 3, scales = "free_y", labeller = as_labeller(ft_labels_1)) +
  ylab(NULL) + theme_pubr(border = TRUE) + theme (legend.title = element_blank ())

# figure6: modeled heat waves - historical heat waves ()
hw_data_1 = read_csv("I:\\1_papers\\6_heat wave variation\\2_data\\7_statistic\\2_modeled_summary\\future-historical_HWE_summary.csv")
hw_data_1$Statistic <- factor(hw_data_1$Statistic, levels = c("Recognition rate", "RMSE (HWF)", "RMSE (HWD)"))
ggscatter(hw_data_1, x = "Year", y = "Value", color = "Statistic", palette = "jco", size = 0.5, add = "loess", conf.int = TRUE) +
  facet_wrap(~GCM, ncol = 5) + ylab(NULL) + theme_pubr(x.text.angle = 45, border = TRUE) + theme (legend.title = element_blank ())

