library(tidyverse)
library(nflfastR)
library(ggthemes)
library(gt)
library(ggridges)

theme_reach <- function() {
  theme_fivethirtyeight() +
    theme(
      legend.position = "none",
      plot.title = element_text(size = 22, hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(size = 16, hjust = 0.5),
      axis.title.x = element_text(size=17),
      axis.title.y = element_text(size=17),
      axis.text = element_text(size = 15)
    )
}

pff_scouting_data <- read_csv("datasets/pffScoutingData.csv")
plays <- read_csv("datasets/plays.csv")
players <- read_csv("datasets/players.csv")
pbp_21 <- load_pbp(2021)
feature_importance <- read_csv("datasets/feature_importance.csv")
sacks_preds <- read_csv("datasets/sacks_preds.csv")

pressure <- pff_scouting_data |> 
  group_by(gameId, playId) |> 
  summarize(pressures = sum(pff_hurry + pff_hit, na.rm = T),
            sacks = sum(pff_sack, na.rm = T),
            hits = sum(pff_hit, na.rm = T),
            hurries = sum(pff_hurry, na.rm = T)) |> 
  mutate(label = case_when(
    pressures == 0 ~ "No Pressure",
    hurries > 0 & hits == 0 & sacks == 0 ~ "Only Hurried",
    hurries == 0 & hits > 0 & sacks == 0 ~ "Only Hit",
    hurries > 0 & hits > 0 & sacks == 0 ~ "Hit & Hurried",
    sacks == 1 ~ "Sacked"
  ))

pressure$gameId <- as.character(pressure$gameId)

pbp_21 |> 
  filter(pass == 1) |> 
  left_join(pressure, by = c("old_game_id" = "gameId", "play_id" = "playId")) |> 
  filter(!is.na(label)) |> 
  group_by(label) |> 
  summarize(count = n(), 
            epa = mean(epa, na.rm = T)) |> 
  mutate(freq = count / sum(count)) |> 
  ggplot(aes(x = fct_reorder(label, -epa), y = epa)) + 
  geom_bar(aes(fill = epa), stat = "identity") +
  scale_fill_viridis_c() +
  labs(x = "Type of Pressure",
       y = "EPA Per Play",
       title = "EPA Per Play Based on Type of Pressure",
       subtitle = "Weeks 1-8 of 2021; frequency listed with each type",
       caption = "Based on Eric Eager's graph on PFF.com") +
  geom_text(aes(label = paste0(round(100*freq, 1), "%"), y = ifelse(epa > 0, -0.05, 0.05)), size = 6) +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 8)) +
  theme_reach() +
  theme(axis.text.x = element_text(size = 16))
ggsave('data-viz/type_of_pressure.png', width = 14, height = 10, dpi = "retina")

feature_importance |> 
  mutate(imp = `...1`, label = `0`) |> 
  ggplot(aes(x = imp, y = fct_reorder(label, imp))) +
  geom_bar(aes(fill = imp), stat = "identity", alpha = 0.8) +
  labs(x = "Importance",
       y = "Label",
       title = "XGBClassifier Feature Importance",
       subtitle="On a scale of 0-1") +
  scale_fill_viridis_c()+
  theme_reach() +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 6)) +
  theme(panel.grid.major.y = element_line(size = 0.01))
ggsave('data-viz/feature-imp-graph.png', width = 14, height = 10, dpi = "retina")

sacks_preds |> 
  mutate(label = ifelse(pff_sack == 1, "Sack", "No Sack")) |> 
  ggplot(aes(x = label, y = sack_pred)) +
  geom_boxplot(aes(fill = label), alpha = 0.9) +
  labs(x = "Label",
       y = "Chance of Sack",
       title = "The Predicted Chance of a Sack Based on the Outcome",
       subtitle = "Outliers are shown in black dots") +
  scale_fill_brewer(palette = "Set1") +
  scale_y_continuous(labels = scales::percent_format(), breaks = scales::pretty_breaks(n = 6)) +
  theme_reach()
ggsave('data-viz/sack-boxplot.png', width = 14, height = 10, dpi = "retina")

sacks_preds |> 
  mutate(label = ifelse(pff_sack == 1, "Sack", "No Sack")) |> 
  ggplot(aes(y = label, x = sack_pred, fill = label)) +
  stat_density_ridges(quantile_lines = TRUE, quantiles = 2) +
  scale_fill_brewer(palette = "Set1") +
  labs(x = "Chance of Sack",
       y = "",
       title = "The Distribution for Chance of Sack Based on Outcome",
       subtitle = "Vertical black line marks the median") +
  scale_x_continuous(labels = scales::percent_format(), breaks = scales::pretty_breaks(n = 6)) +
  theme_reach()
ggsave('data-viz/sack-ridges.png', width = 14, height = 10, dpi = "retina")

headshot <- nflreadr::load_rosters() |> 
  dplyr::select(gsis_it_id, headshot_url)
headshot$gsis_it_id <- as.double(headshot$gsis_it_id)

top_15 <- sacks_preds |> 
  group_by(displayName, nflId, officialPosition) |> 
  summarize(snaps = n(),
            sacks = sum(pff_sack),
            exp_sacks = round(sum(sack_pred, na.rm = T), 2),
            sacks_oe = round(sum(sacks_oe, na.rm = T), 2)) |> 
  arrange(-sacks_oe) |> 
  ungroup() |> 
  mutate(rank = row_number()) |> 
  filter(rank <= 15) |> 
  left_join(headshot, by = c("nflId" = "gsis_it_id")) |> 
  dplyr::select(rank, displayName, headshot_url, snaps, sacks, exp_sacks, sacks_oe) |> 
  gt() |> 
  cols_align(align = "center") |>  
  gtExtras::gt_img_rows(headshot_url) |> 
  gtExtras::gt_theme_538() |> 
  cols_label(rank = "Rank",
             displayName = "Name",
             headshot_url = "",
             snaps = "Snaps",
             sacks = "Sacks",
             exp_sacks = "Expected Sacks",
             sacks_oe = "Sacks Over Expected") |> 
  gtExtras::gt_hulk_col_numeric(sacks_oe) |> 
  tab_header(title = "Sacks Over Expected Leaders",
             subtitle = "Weeks 1-8 of 2021")
gtsave(top_15, "data-viz/top_15.png")



sacks_oe <- sacks_preds |> 
  group_by(displayName, nflId, officialPosition) |> 
  summarize(snaps = n(),
            sacks = sum(pff_sack),
            exp_sacks = sum(sack_pred, na.rm = T),
            sacks_oe = sum(sacks_oe, na.rm = T))



