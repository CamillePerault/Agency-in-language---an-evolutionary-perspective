########## Ce code calcule un score agentif pour les répliques concaténées des personnages haut et bas statut (resp. homme/femme) de chaque pièce de notre corpus ##########
########## et analyse sa corrélation avec lesdits statuts (resp. genres) ##########
########## afin de conclure à la capacité de l'indice linguistique construit à repérer un langage théoriquement plus "agentif" dans un texte ##########

library("dplyr")
library("ggplot2")
library("caTools")
library("broom")
library("tidyr")
library("caret")
library("pROC")

########## STATUT ##########
#####
## préparation des données
# on charge le fichier résultat
df_original_status <- read.csv("C:/Users/camil/OneDrive/Documents/STATUS RESULTS MARCH - CopyR.csv")

## on isole les indicateurs soumis à la contrainte du sujet (qui feront l'objet d'analyses successives non traitées dans ce mémoire)
df_wo_subject_status <- df_original_status[, -grep("with.locutor.as.subject", names(df_original_status))]

# on traite les valeurs manquantes 
rows_with_na_status <- which(is.na(df_wo_subject_status), arr.ind = TRUE)[,1]
df_wo_subject_status[rows_with_na_status,]
colSums(is.na(df_wo_subject_status))

# pour les ratios de type "vs" ratio (pour lesquels beaucoup de valeurs manquent), on remplace les NA par des 0 (en considérant qu'un dénominateur nul est équivalent à un numérateur nul, les ratio étant symétriques)
df_wo_na_vs_status <- df_wo_subject_status
df_wo_na_vs_status[, c("Individual.vs.collective.pronouns", "Definite.vs.undefinite.articles", "Proximal.vs.distal.deictics", "Semantic.modality", "Internal.vs.external.modals", "Individuated.vs.unindividuated.objects", "Active.vs.Passive.Voice", "Perfect.vs.imperfect.tense")] <- replace(df_wo_subject_status[, c("Individual.vs.collective.pronouns", "Definite.vs.undefinite.articles", "Proximal.vs.distal.deictics", "Semantic.modality", "Internal.vs.external.modals", "Individuated.vs.unindividuated.objects", "Active.vs.Passive.Voice", "Perfect.vs.imperfect.tense")], is.na(df_wo_subject_status[, c("Individual.vs.collective.pronouns", "Definite.vs.undefinite.articles", "Proximal.vs.distal.deictics", "Semantic.modality", "Internal.vs.external.modals", "Individuated.vs.unindividuated.objects", "Active.vs.Passive.Voice", "Perfect.vs.imperfect.tense")]), 0)
colSums(is.na(df_wo_na_vs_status))

# pour les ratios non symétriques (numérateur inclus dans le dénominateur) (pour lesquels peu de valeurs manquent), on supprime les NA 
df_wo_na_status <- na.omit(df_wo_na_vs_status)
colSums(is.na(df_wo_na_status))

#on normalise les variables numériques entre 0 et 1
df_numeric_status <- df_wo_na_status %>% select_if(is.numeric) 
df_numeric_status <- df_numeric_status[, -grep("Date", names(df_numeric_status))]
  
df_norm_status <- as.data.frame(apply(df_numeric_status, 2, function(x) { 
  (x - min(x)) / (max(x) - min(x))
  }))

# on ajoute une colonne calculant la somme normalisée pour chaque variable numérique 
df_norm_status <- df_norm_status %>% mutate(sum_normalized_var = rowSums(.))

# on en ajoute une autre qui prend cette fois en compte la direction des prédictions
df_norm_status_dir <- df_norm_status %>% 
  mutate(across(c("Punctuation", "Ellipsis", "Unique.ratio", "Unique.ratio.lemma", "Lexical.richness", "Nouns", "Articles", "Manner.adverbs", "Doubt.adverbs", "Negations", "Negations...verbs", "Present", "Imperfect", "Progressive", "Indicative", "Stative.verbs", "Stative...verbs", "Linking.verbs", "Manner.prep", "Manner.prep.on.adp", "Formality.index", "Dynamic.style"), ~ - .))

# on supprime également les variables pour lesquelles nous n'avions pas d'idée a priori quant à leur corrélation avec l'indice agentif (et qui ont été calculées en préparation d'une analyse successive des marqueurs d'extraversion et d'agréabilité)
df_norm_status_dir <- df_norm_status_dir %>%
  select(-c("Parenthesis", "Dots", "Commas", "Semi.colons", "Colons", "Quotes", "Apostrophes", "Hyphens", "Gerundive")) %>%
  select(-sum_normalized_var)
  
# afin de s'assurer que l'indice ne révèle pas simplement des différences dans la longueur des répliques, on conduit les analyses à l'aide d'une troisième version du score agentif, sans les métriques relatives au nombre total de mots et phrases prononcés  
df_norm_status_woWC <- df_norm_status_dir %>% 
  select(-WC, -Sentence_count)

df_norm_status_dir <- df_norm_status_dir %>% 
  mutate(sum_normalized_var_dir = rowSums(.))
df_norm_status_woWC <- df_norm_status_woWC %>% 
  mutate(sum_normalized_var_woWC = rowSums(.))

# on récupère les variables non numériques (pour des analyses successives)
df_non_numeric_status <- df_wo_na_status %>% select_if(function(x) !is.numeric(x))
df_final_status <- bind_cols(df_non_numeric_status, df_norm_status)
df_final_status_dir <- bind_cols(df_non_numeric_status, df_norm_status_dir)
df_final_statut_woWC <- bind_cols(df_non_numeric_status, df_norm_status_woWC)

#####
## régressions
# on explore et visualise la distribution de la mégavariable ainsi créée, par statut
df_final_status %>% group_by(Status) %>% dplyr::summarize(mean = mean(sum_normalized_var), sd = sd(sum_normalized_var))
df_final_status_dir %>% group_by(Status) %>% dplyr::summarize(mean = mean(sum_normalized_var_dir), sd = sd(sum_normalized_var_dir))
df_final_statut_woWC %>% group_by(Status) %>% dplyr::summarize(mean = mean(sum_normalized_var_woWC), sd = sd(sum_normalized_var_woWC))

# on extrait nos tableaux de régression et on factorise au besoin 
df_catego_status <- df_final_status %>% select(Status, sum_normalized_var)
df_catego_status_dir <- df_final_status_dir %>% select(Status, sum_normalized_var_dir)
df_catego_status_woWC <- df_final_statut_woWC %>% select(Status, sum_normalized_var_woWC)

df_catego_status$sum_normalized_var <- as.numeric(df_catego_status$sum_normalized_var)
df_catego_status_dir$sum_normalized_var_dir <- as.numeric(df_catego_status_dir$sum_normalized_var_dir)
df_catego_status_woWC$sum_normalized_var_woWC <- as.numeric(df_catego_status_woWC$sum_normalized_var_woWC)

# on s'assure de la normalité de la distribution du score agentif pour chaque statut
ggplot(df_catego_status, aes(x = df_catego_status$sum_normalized_var)) + 
  geom_histogram() + 
  facet_wrap(vars(df_catego_status$Status))

ggplot(df_catego_status_dir, aes(x = df_catego_status_dir$sum_normalized_var_dir)) + 
  geom_histogram() + 
  facet_wrap(vars(df_catego_status_dir$Status))

ggplot(df_catego_status_woWC, aes(x = df_catego_status_woWC$sum_normalized_var_woWC)) + 
  geom_histogram() + 
  facet_wrap(vars(df_catego_status_woWC$Status))

# on vérifie qu'il n'y a de fait pas de déséquilibre d'effectifs entre les 2 catégories de statut
xtabs(~ Status, data = df_catego_status)

# on effectue un t.test pour savoir si cette différence de moyennes est significative
t.test(sum_normalized_var ~ Status, data = df_catego_status, alternative = "greater")
t.test(sum_normalized_var_dir ~ Status, data = df_catego_status_dir, alternative = "greater")
t.test(sum_normalized_var_woWC ~ Status, data = df_catego_status_woWC, alternative = "greater")

# on construit un modèle de régression simple expliquant le score de la mégavariable par le statut du personnage
df_catego_status$Status <- factor(df_catego_status$Status)
df_catego_status$Status <- relevel(df_catego_status$Status, ref = "low")

lm_status <- lm(sum_normalized_var ~ Status, data = df_catego_status)
summary(lm_status)

glance(lm_status) %>% pull(sigma)

df_catego_status_dir$Status <- factor(df_catego_status_dir$Status)
df_catego_status_dir$Status <- relevel(df_catego_status_dir$Status, ref = "low")

lm_status_dir <- lm(sum_normalized_var_dir ~ Status, data = df_catego_status_dir)
summary(lm_status_dir)

glance(lm_status_dir) %>% pull(sigma)

df_catego_status_woWC$Status <- factor(df_catego_status_woWC$Status)
df_catego_status_woWC$Status <- relevel(df_catego_status_woWC$Status, ref = "low")

lm_status_woWC <- lm(sum_normalized_var_woWC ~ Status, data = df_catego_status_woWC)
summary(lm_status_woWC)
#toujours significatif !
glance(lm_status_woWC) %>% pull(sigma)

# on visualise les données "inversées" (le statut en fonction de la mégavariable)
df_catego_status_binary <- df_catego_status %>% mutate(Status_binary = ifelse(Status == "low", 0, 1))
df_catego_status_binary$Status_binary <- factor(df_catego_status_binary$Status_binary) 
df_catego_status_dir_binary <- df_catego_status_dir %>% mutate(Status_binary = ifelse(Status == "low", 0, 1))
df_catego_status_dir_binary$Status_binary <- factor(df_catego_status_dir_binary$Status_binary) 
df_catego_status_woWC_binary <- df_catego_status_woWC %>% mutate(Status_binary = ifelse(Status == "low", 0, 1))
df_catego_status_woWC_binary$Status_binary <- factor(df_catego_status_woWC_binary$Status_binary) 

ggplot(df_catego_status_binary, aes(x = df_catego_status_binary$sum_normalized_var, y = df_catego_status_binary$Status_binary)) + geom_point()
ggplot(df_catego_status_dir_binary, aes(x = df_catego_status_dir_binary$sum_normalized_var_dir, y = df_catego_status_dir_binary$Status_binary)) + geom_point()
ggplot(df_catego_status_woWC_binary, aes(x = df_catego_status_woWC_binary$sum_normalized_var_woWC, y = df_catego_status_woWC_binary$Status_binary)) + geom_point()

# on construit un modèle de régression logit, expliquant à l'inverse le statut par le score de la mégavariable
set.seed(123)
trainIndex <- createDataPartition(df_catego_status_binary$Status_binary, p = 0.7, 
                                  list = FALSE,
                                  times = 1)
train_data <- df_catego_status_binary[trainIndex, ]
test_data <- df_catego_status_binary[-trainIndex, ]

glm_status_reversed <- glm(Status_binary ~ sum_normalized_var, family = binomial, data = train_data)
summary(glm_status_reversed)

# on calcule les odds-ratios
coefficients <- coef(glm_status_reversed)
odds_ratios <- exp(coefficients)

conf_int <- confint(glm_status_reversed)
odds_ratios_CI <- exp(conf_int)

data.frame(
  Predictor = names(odds_ratios),
  Odds_Ratio = odds_ratios,
  Lower_CI = odds_ratios_CI[, 1],
  Upper_CI = odds_ratios_CI[, 2]
)

# on construit le même modèle logit avec cette fois la prise en compte des directions de corrélation
trainIndex_dir <- createDataPartition(df_catego_status_dir_binary$Status_binary, p = 0.7, 
                                      list = FALSE,
                                      times = 1)
train_data_dir <- df_catego_status_dir_binary[trainIndex_dir, ]
test_data_dir <- df_catego_status_dir_binary[-trainIndex_dir, ]

glm_status_dir_reversed <- glm(Status_binary ~ sum_normalized_var_dir, family = binomial, data = train_data_dir)
summary(glm_status_dir_reversed)

# on calcule les odds-ratio
coefficients_dir <- coef(glm_status_dir_reversed)
odds_ratios_dir <- exp(coefficients_dir)

conf_int_dir <- confint(glm_status_dir_reversed)
odds_ratios_CI_dir <- exp(conf_int_dir)

data.frame(
  Predictor = names(odds_ratios_dir),
  Odds_Ratio = odds_ratios_dir,
  Lower_CI = odds_ratios_CI_dir[, 1],
  Upper_CI = odds_ratios_CI_dir[, 2]
)

# idem pour la version 3
trainIndex_woWC <- createDataPartition(df_catego_status_woWC_binary$Status_binary, p = 0.7, 
                                      list = FALSE,
                                      times = 1)
train_data_woWC <- df_catego_status_woWC_binary[trainIndex_woWC, ]
test_data_woWC <- df_catego_status_woWC_binary[-trainIndex_woWC, ]

glm_status_woWC_reversed <- glm(Status_binary ~ sum_normalized_var_woWC, family = binomial, data = train_data_woWC)
summary(glm_status_woWC_reversed)

coefficients_woWC <- coef(glm_status_woWC_reversed)
odds_ratios_woWC <- exp(coefficients_woWC)

conf_int_woWC <- confint(glm_status_woWC_reversed)
odds_ratios_CI_woWC <- exp(conf_int_woWC)

data.frame(
  Predictor = names(odds_ratios_woWC),
  Odds_Ratio = odds_ratios_woWC,
  Lower_CI = odds_ratios_CI_woWC[, 1],
  Upper_CI = odds_ratios_CI_woWC[, 2]
)

# on évalue leurs performances prédictives des différents modèles
set.seed(123)
control <- trainControl(method = "cv", number = 10)

cv_model <- train(Status_binary ~ sum_normalized_var, data = train_data, method = "glm", family = "binomial", trControl = control)
cv_predictions <- predict(cv_model, newdata = test_data, type = "raw")
cv_metrics <- confusionMatrix(cv_predictions, test_data$Status_binary)

confusion_matrix <- table(Actual = test_data$Status_binary, Predicted = cv_predictions)
print(confusion_matrix)

precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)

cv_model_dir <- train(Status_binary ~ sum_normalized_var_dir, data = train_data_dir, method = "glm", family = "binomial", trControl = control)
cv_predictions_dir <- predict(cv_model_dir, newdata = test_data_dir, type = "raw")
cv_metrics_dir <- confusionMatrix(cv_predictions_dir, test_data_dir$Status_binary)

confusion_matrix_dir <- table(Actual = test_data_dir$Status_binary, Predicted = cv_predictions_dir)
print(confusion_matrix_dir)

precision_dir <- confusion_matrix_dir[2, 2] / sum(confusion_matrix_dir[, 2])
recall_dir <- confusion_matrix_dir[2, 2] / sum(confusion_matrix_dir[2, ])
f1_score_dir <- 2 * (precision_dir * recall_dir) / (precision_dir + recall_dir)

cv_model_woWC <- train(Status_binary ~ sum_normalized_var_woWC, data = train_data_woWC, method = "glm", family = "binomial", trControl = control)
cv_predictions_woWC <- predict(cv_model_woWC, newdata = test_data_woWC, type = "raw")
cv_metrics_woWC <- confusionMatrix(cv_predictions_woWC, test_data_woWC$Status_binary)

confusion_matrix_woWC <- table(Actual = test_data_woWC$Status_binary, Predicted = cv_predictions_woWC)
print(confusion_matrix_woWC)

precision_woWC <- confusion_matrix_woWC[2, 2] / sum(confusion_matrix_woWC[, 2])
recall_woWC <- confusion_matrix_woWC[2, 2] / sum(confusion_matrix_woWC[2, ])
f1_score_dir <- 2 * (precision_woWC * recall_woWC) / (precision_woWC + recall_woWC)

# analyse exploratoire de l'effet de l'année sur l'association entre statut et score agentif
df_catego_status_date <- cbind(df_catego_status_dir, Date = df_wo_na_status$Date)
df_catego_status_date$Status <- relevel(df_catego_status_date$Status, ref = "high")
length(unique(df_catego_status_date$Date)) #156 années distinctes

library(Hmisc)
ggplot(df_catego_status_date, aes(x = Date, y = sum_normalized_var_dir, color = Status)) +
  stat_summary(
    aes(group = Status),
    fun.data = mean_sdl,
    fun.args = list(mult = 1),  
    geom = "line"
  ) +
  labs(x = "Année de Publication", y = "Score agentif moyen") +
  scale_color_manual(values = c("high" = "blue", "low" = "red")) +
  theme_minimal()
# présence de valeurs extrêmes à la fin de la période + valeurs beaucoup plus espacées à partir de 1807

# on tronque donc la fin de la période (1808-1885)
df_catego_status_date <- df_catego_status_date %>%
  filter(!(Date>=1808))

ggplot(df_catego_status_date, aes(x = Date, y = sum_normalized_var_dir, color = Status)) +
  stat_summary(
    aes(group = Status),
    fun.data = mean_sdl,
    fun.args = list(mult = 1),  
    geom = "line"
  ) +
  labs(x = "Année de Publication", y = "Score agentif moyen", color = "Statut") +
  scale_color_manual(values = c("high" = "red", "low" = "blue"), 
                     labels = c("high" = "haut", "low" = "bas") ) +
  theme_minimal() +
  theme(
    axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 20, l = 0)), 
    axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)))
    
  
# on calcule l'ICC de la variable date à l'aide d'un modèle hiérarchique
library(lme4)
lmer_status_date <- lmer(sum_normalized_var_dir ~ Status + (1 | Date), data = df_catego_status_date)
summary(lmer_status_date)
var_components <- VarCorr(lmer_status_date)
year_variance <- var_components$Date
total_variance <- sigma(lmer_status_date)^2
icc_value <- year_variance / (year_variance + total_variance)
#10%

# on agrège les différentes dates en 2 intervalles (1 par siècle), afin de faire tourner un modèle linéaire simple sur chacune de ces périodes et comparer les coefficients obtenus
df_catego_status_date <- df_catego_status_date[order(df_catego_status_date$Date), ]
interval_bounds <- c(1629, 1700)
df_catego_status_date$Interval <- factor(
  findInterval(df_catego_status_date$Date, interval_bounds))

regression_results <- list()

for (interval_name in unique(df_catego_status_date$Interval)) {
  interval_data <- subset(df_catego_status_date, Interval == interval_name)
  lm_result <- lm(sum_normalized_var_dir ~ Status, data = interval_data)
  print(summary(lm_result))
  regression_results[[interval_name]] <- lm_result
}

########## GENRE ##########
#####
## préparation des données
# on charge le fichier résultat
df_original_gender <- read.csv("C:/Users/camil/OneDrive/Documents/GENDER RESULTS MARCH - CopyR.csv")

# on isole les indicateurs sousmis à la contrainte du sujet (qui feront l'objet d'analyses successives non traitées dans ce rapport)
df_subject_gender  <- df_original_gender[, grep("with.locutor.as.subject", names(df_original_gender))]
df_wo_subject_gender <- df_original_gender[, -grep("with.locutor.as.subject", names(df_original_gender))]

# on traite les valeurs manquantes 
rows_with_na_gender <- which(is.na(df_wo_subject_gender), arr.ind = TRUE)[,1]
df_wo_subject_gender[rows_with_na_gender,]
colSums(is.na(df_wo_subject_gender))

# pour les ratios de type "vs" ratio (pour lesquels beaucoup de valeurs manquent), on remplace les NA par des 0 (en considérant qu'un dénominateur nul est équivalent à un numérateur nul, les ratio étant symétriques)
df_wo_na_vs_gender <- df_wo_subject_gender
df_wo_na_vs_gender[, c("Individual.vs.collective.pronouns", "Definite.vs.undefinite.articles", "Proximal.vs.distal.deictics", "Semantic.modality", "Internal.vs.external.modals", "Individuated.vs.unindividuated.objects", "Active.vs.Passive.Voice", "Perfect.vs.imperfect.tense")] <- replace(df_wo_subject_gender[, c("Individual.vs.collective.pronouns", "Definite.vs.undefinite.articles", "Proximal.vs.distal.deictics", "Semantic.modality", "Internal.vs.external.modals", "Individuated.vs.unindividuated.objects", "Active.vs.Passive.Voice", "Perfect.vs.imperfect.tense")], is.na(df_wo_subject_gender[, c("Individual.vs.collective.pronouns", "Definite.vs.undefinite.articles", "Proximal.vs.distal.deictics", "Semantic.modality", "Internal.vs.external.modals", "Individuated.vs.unindividuated.objects", "Active.vs.Passive.Voice", "Perfect.vs.imperfect.tense")]), 0)
colSums(is.na(df_wo_na_vs_gender))

# pour les ratios non symétriques (numérateur inclus dans le dénominateur) (pour lesquels peu de valeurs manquent), on supprime les NA 
df_wo_na_gender <- na.omit(df_wo_na_vs_gender)
colSums(is.na(df_wo_na_gender))

# on normalise les variables numériques entre 0 et 1
df_numeric_gender <- df_wo_na_gender %>% select_if(is.numeric) 

df_norm_gender <- as.data.frame(apply(df_numeric_gender, 2, function(x) { 
  (x - min(x)) / (max(x) - min(x))
}))

# on ajoute une colonne calculant la somme normalisée pour chaque variable numérique 
df_norm_gender <- df_norm_gender %>% 
  mutate(sum_normalized_var = rowSums(.))

# cette fois,  nous n'avons pas d'idée arrêtée sur les directions attendues (score augmentant ou non selon que le genre est féminin ou masculin) pour les métriques prises individuellement
df_norm_gender_dir <- df_norm_gender %>% 
  mutate(across(c("Punctuation", "Ellipsis", "Unique.ratio", "Unique.ratio.lemma", "Lexical.richness", "Nouns", "Articles", "Manner.adverbs", "Doubt.adverbs", "Negations", "Negations...verbs", "Present", "Imperfect", "Progressive", "Indicative", "Stative.verbs", "Stative...verbs", "Linking.verbs", "Manner.prep", "Manner.prep.on.adp", "Formality.index", "Dynamic.style"), ~ - .)) %>%
  select(-c("Parenthesis", "Dots", "Commas", "Semi.colons", "Colons", "Quotes", "Apostrophes", "Hyphens", "Gerundive")) %>%
  select(-sum_normalized_var) %>% 
  mutate(sum_normalized_var_dir = rowSums(.))

# on récupère les variables non numériques
df_non_numeric_gender <- df_wo_na_gender %>% select_if(function(x) !is.numeric(x))
df_final_gender <- bind_cols(df_non_numeric_gender, df_norm_gender)
df_final_gender_dir <- bind_cols(df_non_numeric_gender, df_norm_gender_dir)

######
## régressions
# on explore et visualise la distribution de la mégavariable ainsi créée, par genre
df_final_gender %>% group_by(Gender) %>% dplyr::summarize(mean = mean(sum_normalized_var), sd = sd(sum_normalized_var))
df_catego_gender <- df_final_gender %>% select(Gender, sum_normalized_var)

df_final_gender_dir %>% group_by(Gender) %>% dplyr::summarize(mean = mean(sum_normalized_var_dir), sd = sd(sum_normalized_var_dir))
df_catego_gender_dir <- df_final_gender_dir %>% select(Gender, sum_normalized_var_dir)

# on vérifie la normalité de la distribution
ggplot(df_catego_gender, aes(x = df_catego_gender$sum_normalized_var)) + geom_histogram() + facet_wrap(vars(df_catego_gender$Gender))
ggplot(df_catego_gender_dir, aes(x = df_catego_gender_dir$sum_normalized_var_dir)) + geom_histogram() + facet_wrap(vars(df_catego_gender_dir$Gender))

# on vérifie qu'il n'y a de fait pas de déséquilibre d'effectifs entre les 2 catégories de genre
xtabs(~ Gender, data = df_catego_gender)

# on effectue un t.test pour savoir si cette différence de moyennes est significative
t.test(sum_normalized_var ~ Gender, data = df_catego_gender, alternative = "greater")
t.test(sum_normalized_var_dir ~ Gender, data = df_catego_gender_dir, alternative = "greater")

# on construit un modèle de régression simple expliquant le score de la mégavariable par le genre du personnage
df_catego_gender$Gender <- factor(df_catego_gender$Gender)
df_catego_gender$sum_normalized_var <- as.numeric(df_catego_gender$sum_normalized_var)

df_catego_gender_dir$Gender <- factor(df_catego_gender_dir$Gender)
df_catego_gender_dir$sum_normalized_var_dir <- as.numeric(df_catego_gender_dir$sum_normalized_var_dir)

lm_gender <- lm(sum_normalized_var ~ Gender, data = df_catego_gender)
summary(lm_gender)

glance(lm_gender) %>% pull(sigma)

lm_gender_dir <- lm(sum_normalized_var_dir ~ Gender, data = df_catego_gender_dir)
summary(lm_gender_dir) #non significatif

glance(lm_gender_dir) %>% pull(sigma)

# on visualise les données "inversées" (le statut en fonction de la mégavariable)
df_catego_gender_binary <- df_catego_gender %>% mutate(Gender_binary = ifelse(Gender == "female", 0, 1))
df_catego_gender_binary$Gender_binary <- factor(df_catego_gender_binary$Gender_binary)

ggplot_gender_reversed <- ggplot(df_catego_gender_binary, aes(x = df_catego_gender_binary$sum_normalized_var, y = df_catego_gender_binary$Gender_binary)) + geom_point()
ggplot_gender_reversed

df_catego_gender_dir_binary <- df_catego_gender_dir %>% mutate(Gender_binary = ifelse(Gender == "female", 0, 1))
df_catego_gender_dir_binary$Gender_binary <- factor(df_catego_gender_dir_binary$Gender_binary)

ggplot_gender_reversed_dir <- ggplot(df_catego_gender_dir_binary, aes(x = df_catego_gender_dir_binary$sum_normalized_var_dir, y = df_catego_gender_dir_binary$Gender_binary)) + geom_point()
ggplot_gender_reversed_dir

# on construit un modèle de régression logit, expliquant à l'inverse le genre par le score de la mégavariable
set.seed(123)

trainIndex <- createDataPartition(df_catego_gender_binary$Gender_binary, p = 0.7, 
                                  list = FALSE,
                                  times = 1)
train_data_gender <- df_catego_gender_binary[trainIndex, ]
test_data_gender <- df_catego_gender_binary[-trainIndex, ]

glm_gender_reversed <- glm(Gender_binary ~ sum_normalized_var, family = binomial, data = train_data_gender)
summary(glm_gender_reversed)

trainIndex_dir <- createDataPartition(df_catego_gender_dir_binary$Gender_binary, p = 0.7, 
                                      list = FALSE,
                                      times = 1)
train_data_gender_dir <- df_catego_gender_dir_binary[trainIndex_dir, ]
test_data_gender_dir <- df_catego_gender_dir_binary[-trainIndex_dir, ]

glm_gender_reversed_dir <- glm(Gender_binary ~ sum_normalized_var_dir, family = binomial, data = train_data_gender_dir)
summary(glm_gender_reversed_dir)

# on calcule les odds-ratio
coefficients_gender <- coef(glm_gender_reversed)
odds_ratios_gender <- exp(coefficients_gender)

conf_int_gender <- confint(glm_gender_reversed)
odds_ratios_CI_gender <- exp(conf_int_gender)

data.frame(
  Predictor = names(odds_ratios_gender),
  Odds_Ratio = odds_ratios_gender,
  Lower_CI = odds_ratios_CI_gender[, 1],
  Upper_CI = odds_ratios_CI_gender[, 2]
)

coefficients_gender_dir <- coef(glm_gender_reversed_dir)
odds_ratios_gender_dir <- exp(coefficients_gender_dir)

conf_int_gender_dir <- confint(glm_gender_reversed_dir)
odds_ratios_CI_gender_dir <- exp(conf_int_gender_dir)

data.frame(
  Predictor = names(odds_ratios_gender_dir),
  Odds_Ratio = odds_ratios_gender_dir,
  Lower_CI = odds_ratios_CI_gender_dir[, 1],
  Upper_CI = odds_ratios_CI_gender_dir[, 2]
)

# on évalue ses performances prédictives
set.seed(123)
control <- trainControl(method = "cv", number = 10)

cv_model_gender <- train(Gender_binary ~ sum_normalized_var, data = train_data_gender, method = "glm", family = "binomial", trControl = control)
cv_predictions_gender <- predict(cv_model_gender, newdata = test_data_gender, type = "raw")
cv_metrics_gender <- confusionMatrix(cv_predictions_gender, test_data_gender$Gender_binary)

confusion_matrix_gender <- table(Actual = test_data_gender$Gender_binary, Predicted = cv_predictions_gender)
print(confusion_matrix_gender)

precision_gender <- confusion_matrix_gender[2, 2] / sum(confusion_matrix_gender[, 2])
recall_gender <- confusion_matrix_gender[2, 2] / sum(confusion_matrix_gender[2, ])
f1_score_gender <- 2 * (precision_gender * recall_gender) / (precision_gender + recall_gender)

cv_model_gender_dir <- train(Gender_binary ~ sum_normalized_var_dir, data = train_data_gender_dir, method = "glm", family = "binomial", trControl = control)
cv_predictions_gender_dir <- predict(cv_model_gender_dir, newdata = test_data_gender_dir, type = "raw")
cv_metrics_gender_dir <- confusionMatrix(cv_predictions_gender_dir, test_data_gender_dir$Gender_binary)

confusion_matrix_gender_dir <- table(Actual = test_data_gender_dir$Gender_binary, Predicted = cv_predictions_gender_dir)
print(confusion_matrix_gender_dir)

precision_gender_dir <- confusion_matrix_gender_dir[2, 2] / sum(confusion_matrix_gender_dir[, 2])
recall_gender_dir <- confusion_matrix_gender_dir[2, 2] / sum(confusion_matrix_gender_dir[2, ])
f1_score_gender_dir <- 2 * (precision_gender_dir * recall_gender_dir) / (precision_gender_dir + recall_gender_dir)
#le modèle avec prise en compte des directions ne performe de fait pas mieux, et se révèle même décidément non significatif là où le premier modèle l'est faiblement
