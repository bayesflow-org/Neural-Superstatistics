library(tidyverse)
library(magrittr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

files <- fs::dir_ls("raw_data/")
tb <- read_tsv(files, id = "path")

summary <- tb %>% 
  group_by(path) %>% 
  summarise(acc = mean(correct))

tb_subset <- tb %>% 
  filter(path %in% summary$path[which(summary$acc > 0.7)])

meta <- do.call("rbind",strsplit(sub("\\."," ", tb_subset$path),"_"))

tb_subset %<>%
  mutate(id = extract_numeric(meta[, 3]),
         feedback = meta[, 4],
         trial_condition = meta[, 5]) %>% 
  relocate(id, feedback, trial_condition) %>% 
  select(-c(path, coherentDots, numberofDots,
            percentCoherence, eventCount, averageFrameRate)) %>% 
  rename(block = blkNum,
         trial = trlNum,
         correct_resp = winningDirection,
         resp = response,
         rt = RT) %>% 
  mutate(rt = ifelse(correct == 1, rt/1000, -(rt/1000)))

summary <- tb_subset %>% 
  group_by(id) %>% 
  summarise(n = n())

write_csv(tb_subset, "optimal_policy_data.csv")