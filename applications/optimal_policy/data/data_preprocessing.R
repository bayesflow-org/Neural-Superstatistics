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
  mutate(id = extract_numeric(meta[, 2]),
         feedback = meta[, 3],
         trial_condition = meta[, 4]) %>% 
  relocate(id, feedback, trial_condition) %>% 
  select(-c(path, coherentDots, numberofDots,
            percentCoherence, eventCount, averageFrameRate)) %>% 
  rename(block = blkNum,
         trial = trlNum,
         correct_resp = winningDirection,
         resp = response,
         rt = RT) %>% 
  mutate(rt = ifelse(correct == 1, rt/1000, -(rt/1000)),
         id = factor(id,
                     levels = unique(id),
                     labels = seq(1, length(unique(id)))),
         id = as.numeric(as.character(id)))

write_csv(tb_subset, "optimal_policy_data.csv")





