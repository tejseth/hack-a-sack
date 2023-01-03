library(tidyverse)
library(nflfastR)
library(ggthemes)
library(gt)
library(ggridges)
library(ggimage)

theme_reach <- function() {
  theme_fivethirtyeight() +
    theme(
      legend.position = "none",
      plot.title = element_text(size = 24, hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(size = 18, hjust = 0.5),
      axis.title.x = element_text(size=20),
      axis.title.y = element_text(size=20),
      axis.text = element_text(size = 17)
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
  theme_reach()
ggsave('data-viz/type_of_pressure.png', width = 14, height = 10, dpi = "retina")

feature_importance |> 
  mutate(imp = `...1`, label = `0`) |> 
  filter(imp > 0.01) |> 
  ggplot(aes(x = imp, y = fct_reorder(label, imp))) +
  geom_bar(aes(fill = imp), stat = "identity", alpha = 0.8) +
  labs(x = "Importance",
       y = "Label",
       title = "XGBClassifier Feature Importance",
       subtitle="Features with a 0.01 importance or greater shown (on a scale of 0-1)") +
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

headshot <- nflreadr::load_rosters(2021) |> 
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

defenses <- plays |> 
  dplyr::select(gameId, playId, defensiveTeam)

sacks_oe <- sacks_preds |> 
  left_join(defenses, by = c("gameId", "playId")) |> 
  group_by(displayName, nflId, officialPosition) |> 
  summarize(snaps = n(),
            team = first(defensiveTeam),
            sacks = mean(pff_sack),
            exp_sacks = mean(sack_pred, na.rm = T),
            sacks_oe = sum(sacks_oe, na.rm = T)) |> 
  filter(snaps >= 200) |> 
  left_join(teams_colors_logos, by = c("team" = "team_abbr"))

sacks_oe |> 
  ggplot(aes(x = exp_sacks, y = sacks)) + 
  geom_hline(yintercept = mean(sacks_oe$sacks), linetype = "dashed") +
  geom_vline(xintercept = mean(sacks_oe$exp_sacks), linetype = "dashed") +
  geom_point(aes(fill = team_color2, color = team_color, size = snaps), 
             alpha = 0.9, shape = 21) +
  scale_color_identity(aesthetics = c("fill", "color")) +
  geom_smooth(method = "lm", se = FALSE, color = "gray", size = 1.5) +
  ggrepel::geom_text_repel(aes(label = displayName), size = 5) +
  labs(x = "Expected Sacks Per Snap",
       y = "Sacks Per Snap",
       title = "How Expected Sacks and Sacks are Correlated",
       subtitle = "Minimum of 200 snaps in weeks 1-8 of 2021, bubble size is amount of snaps") +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 8)) +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 8)) +
  theme_reach()
ggsave('data-viz/exp-actual-sacks.png', width = 14, height = 10, dpi = "retina")

team_sacks_oe <- sacks_preds |> 
  left_join(defenses, by = c("gameId", "playId")) |> 
  group_by(defensiveTeam) |> 
  summarize(sacks = sum(pff_sack),
            exp_sacks = sum(sack_pred, na.rm = T),
            sacks_oe = sum(sacks_oe, na.rm = T)) |> 
  left_join(teams_colors_logos, by = c("defensiveTeam" = "team_abbr"))

team_sacks_oe |> 
  ggplot(aes(x = exp_sacks, y = sacks)) +
  geom_hline(yintercept = mean(team_sacks_oe$sacks), linetype = "dashed", alpha = 0.5) +
  geom_vline(xintercept = mean(team_sacks_oe$exp_sacks), linetype = "dashed", alpha = 0.5) +
  geom_image(aes(image = team_logo_espn), asp = 16/9, size = 0.05) +
  geom_smooth(method = "lm", se = FALSE, color = "gray", size = 1.5) +
  labs(x = "Expected Sacks",
       y = "Actual Sacks",
       title = "Expected Sacks and Actual Sacks on the Team Level",
       subtitle = "Weeks 1-8 of the 2021 season") +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 8)) +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 8)) +
  theme_reach()
ggsave('data-viz/team-sacks.png', width = 14, height = 10, dpi = "retina")

calibration <- sacks_preds |> 
  filter(!is.na(sack_pred)) |> 
  mutate(
    bin_pred_prob = round(sack_pred / 0.01) * .01,
  ) |> 
  group_by(bin_pred_prob) |> 
  summarize(
    n_plays = n(),
    n_sack = length(which(pff_sack == 1)),
    bin_actual_prob = n_sack / n_plays
  )

weeks <- pbp_21 |> 
  dplyr::select(old_game_id, play_id, week)

weeks$old_game_id <- as.double(weeks$old_game_id)

stability <- sacks_preds |> 
  left_join(weeks, by = c("gameId" = "old_game_id", "playId" = "play_id")) |> 
  left_join(defenses, by = c("gameId", "playId")) |> 
  mutate(week_label = ifelse(week < 5, "weeks_1_4", "weeks_5_8")) |> 
  group_by(displayName, week_label) |> 
  summarize(snaps = n(),
            team = first(defensiveTeam),
            sacks = mean(pff_sack),
            exp_sacks = mean(sack_pred, na.rm = T),
            sacks_oe = sum(sacks_oe, na.rm = T)) |> 
  pivot_wider(
    id_cols = c(displayName, team),
    names_from = week_label,
    values_from = c(snaps, sacks, exp_sacks, sacks_oe)) |> 
  mutate(total_snaps = snaps_weeks_1_4 + snaps_weeks_5_8) |> 
  filter(total_snaps >= 200) |> 
  left_join(teams_colors_logos, by = c("team" = "team_abbr"))

summary(lm(sacks_weeks_5_8 ~ sacks_weeks_1_4, data = stability))$r.squared # 0.41
summary(lm(exp_sacks_weeks_5_8 ~ exp_sacks_weeks_1_4, data = stability))$r.squared # 0.00
summary(lm(sacks_oe_weeks_5_8 ~ sacks_oe_weeks_1_4, data = stability))$r.squared # 0.24

stability |> 
  ggplot(aes(x = sacks_oe_weeks_1_4, y = sacks_oe_weeks_5_8)) +
  geom_hline(yintercept = mean(stability$sacks_oe_weeks_5_8), linetype = "dashed") +
  geom_vline(xintercept = mean(stability$sacks_oe_weeks_1_4), linetype = "dashed") +
  geom_point(aes(fill = team_color2, color = team_color), 
             alpha = 0.9, shape = 21, size = 4) +
  scale_color_identity(aesthetics = c("fill", "color")) +
  geom_smooth(method = "lm", se = FALSE, color = "gray", size = 1.5) +
  ggrepel::geom_text_repel(aes(label = displayName), size = 5, max.overlaps = 3) +
  labs(x = "Sacks Over Expected Weeks 1-4",
       y = "Sacks Over Expected Weeks 5-8",
       title = "How Stable Sacks Over Expected are From Weeks 1-4 to Weeks 5-8",
       subtitle = "Minimum of 200 snaps in weeks 1-8 of 2021, bubble size is amount of snaps") +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 8)) +
  scale_y_continuous(breaks = scales::pretty_breaks(n = 8)) +
  theme_reach()
ggsave('data-viz/stable.png', width = 14, height = 10, dpi = "retina")


