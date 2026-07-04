# TFM UNIR — Predicción de Resultados en Carreras de Caballos

**TFM — Máster en Ingeniería de Inteligencia Artificial · UNIR**  
**Autor:** Daniel Ojados  
**Datos:** Real Sociedad Hípica Española · circuito español · hasta abril de 2025  

---

## Descripción del Proyecto

A continuación el código del Trabajo de Fin de Máster en Ingeniería de Inteligencia Artificial. El objetivo es predecir qué caballos terminarán en el **podio (top-3)** de una carrera del circuito español, formulando el problema como **ranking tabular** (*Learning-to-Rank*).

El proyecto compara técnicas clásicas de *boosting* (XGBoost, LightGBM, CatBoost) con arquitecturas tabulares profundas más recientes (TabM, TabTransformer), además de rankers directos (LGBMRanker, XGBRanker) y baselines de referencia. La evaluación se basa en métricas de ranking orientadas a lista (NDCG@3, MAP@3) calculadas sobre un conjunto de test cronológicamente posterior al entrenamiento.

---

## Estructura del Repositorio

```
bucephalus/
│
├── notebooks/
│   ├── 00_pipeline_fuentes.ipynb                  # Pipeline de datos y feature store
│   ├── 01_eda_calidad_fiabilidad.ipynb            # EDA, auditoría y preprocesamiento
│   └── 02_modelado_cientifico.ipynb               # Modelado, evaluación e interpretabilidad
│
├── pipelines/                                     # Scripts Python del pipeline canónico
│   ├── 00_audit_and_clean.py                      # Limpieza y auditoría de CSVs raw
│   ├── 01_validate_relations.py                   # Validación de claves foráneas
│   ├── 02_build_master_dataset.py                 # Construcción del master canónico
│   └── 04_feature_engineering.py                  # Feature engineering + meteorología
│
├── data/
│   ├── raw/                                       # Ficheros fuente (no incluidos)
│   └── processed/
│       └── master_v5_final.parquet                # Dataset final (no incluido)
│
├── models/                                        # Artefactos de preprocesamiento y modelos
│   ├── imputer.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_meta.json
│
└── reports/                                       # Resultados y métricas exportadas
    └── model_results_test_official.csv
```

---

## Flujo de Trabajo

El pipeline se ejecuta en tres pasos secuenciales. Cada notebook toma como entrada los artefactos del anterior.

```
Fuentes raw (Real Sociedad Hípica Española + Open-Meteo)
        │
        ▼
[00] Pipeline de datos
        │  → master_v5_final.parquet (101.976 filas × 215 columnas)
        ▼
[01] EDA, auditoría y preprocesamiento
        │  → imputer.pkl, scaler.pkl, label_encoders.pkl, feature_meta.json
        ▼
[02] Modelado y evaluación
        │  → model_results_test_official.csv, test_predictions_official.npz
        ▼
Resultados: LightGBM_HPO  →  NDCG@3 = 0,6055 (test)
```

---

## Notebooks

### `00_pipeline_fuentes.ipynb` — Pipeline de Datos y Feature Store

Este notebook transforma las nueve fuentes de datos heterogéneas en un único dataset tabular listo para el modelado. La unidad de análisis es **una fila por participante en una carrera real**.

#### Fuentes de datos

Los datos proceden del portal oficial de la Real Sociedad Hípica Española, extraídos mediante scraping hasta el 29 de marzo de 2025. Las fuentes se leen desde Google Cloud Storage o, en su defecto, desde disco local.

| Fichero fuente | Entidad | Clave principal | Relación |
|---|---|---|---|
| `jornadas_delta_20250329.csv` | Jornada (meeting) | `jornada_id` | FK → carreras |
| `carreras_delta_20250329.csv` | Carrera | `carrera_id` | FK → participantes, jornadas |
| `participantes_merged_20250329.csv` | Caballo × Carrera | `(carrera_id, caballo_id)` | Tabla estrella |
| `caballos_historial_delta_20250329.csv` | Caballo × Carrera | `(caballo_id, fecha)` | Historial de actuaciones |
| `jinetes_info_delta_20250329.csv` | Jinete | `jinete_id` | FK → historial |
| `jinetes_historial_delta_20250329.csv` | Jinete × Carrera | `(jinete_id, fecha)` | Historial de actuaciones |
| `preparadores_info_delta_20250329.csv` | Preparador | `preparador_id` | FK → historial |
| `preparadores_historial_delta_20250329.csv` | Preparador × Carrera | `(preparador_id, fecha)` | Historial de actuaciones |
| `caballos_con_ids_progenitores.xlsx` | Genealogía | `caballo_id` | Padre, madre, linaje |

La meteorología histórica se obtiene de la **API Open-Meteo** para cada hipódromo y fecha de carrera.

#### Pipeline canónico (scripts)

La reproducción completa desde cero se realiza ejecutando los scripts de `pipelines/` en orden:

```
PASO 1 — pipelines/00_audit_and_clean.py
         Limpieza de CSVs raw: tipos, fechas, duplicados, rangos.

PASO 2 — pipelines/01_validate_relations.py
         Validación de integridad referencial entre tablas.

PASO 3 — pipelines/02_build_master_dataset.py
         Construcción del master canónico (sin carreras sintéticas).
         Resultado: tabla estrella con clave (carrera_id, caballo_id).

PASO 4 — pipelines/04_feature_engineering.py
         Feature engineering completo + variables delta + meteorología.
         Resultado: master_v5_final.parquet (215 columnas).
```

Si ya se dispone del parquet procesado, los scripts pueden omitirse y el notebook lee directamente desde BigQuery o desde disco.

#### Feature Store — Familias de Variables

El dataset final contiene **215 columnas** (115 documentadas en el catálogo de características más variables auxiliares e identificadores). Las variables se organizan en 14 familias:

| Familia | Variables | Descripción |
|---|---|---|
| Contexto de carrera | 13 | Hipódromo, distancia, superficie, número de participantes, premio |
| Hándicap pre-carrera | 8 | Peso asignado, casilla de salida, posición relativa del cajón |
| Forma del caballo | 14 | Win rate acumulado, medias móviles (EWM), experiencia, posición media reciente |
| Forma del jinete | 13 | Win rate, top-3 rate, medias móviles, número de carreras recientes |
| Forma del preparador | 10 | Win rate, top-3 rate, medias móviles, experiencia en distancia |
| Sinergias caballo-jinete / caballo-preparador | 4 | Tasa de éxito conjunta en el historial compartido |
| Aptitud por distancia / hipódromo / superficie | 9 | Win rate específico por contexto de carrera |
| Variables intra-carrera (within-race) | 5 | Z-scores y rankings relativos al campo de esa carrera |
| Campo / competencia | 7 | Calidad media del campo, percentiles de forma |
| Genealogía | 5 | Tasa de éxito del padre y la madre, especialización heredada |
| Variables delta | 16 | Cambios respecto a la carrera anterior (distancia, jinete, hipódromo) |
| Mercado / cuotas | 2 | Probabilidad implícita de la cuota de victoria |
| Meteorología | 6 | Temperatura, lluvia, viento, estado del terreno en el momento de la carrera |
| Targets | 3 | `target_top1`, `target_top3`, `target_pos` (posición final) |

#### Prevención de Leakage Temporal

Todas las variables históricas se calculan aplicando `shift(1)` antes de cualquier ventana móvil o acumulada, garantizando que la fila de la carrera `r_i` solo contiene información de las carreras `r_0, ..., r_{i-1}`. El notebook implementa ocho verificaciones explícitas:

| Check | Descripción |
|---|---|
| L1 | `shift(1)` aplicado: `horse_n_prev_races == 0` en la primera aparición de cada caballo |
| L2 | Genealogía: sin variabilidad intradía por semental (padre/madre no cambian en el día) |
| L3 | Variables target (`target_*`) excluidas del feature set |
| L4 | `odds_win` con correlación negativa con el target (señal válida, no leakage) |
| L5 | Meteorología: temperatura y lluvia disponibles antes de la salida |
| L6 | Variables post-carrera (`pos`, `distancia_al_anterior`, `race_has_winner`) fuera del feature set |
| L7 | Estabilidad temporal del poder predictivo (2005–2024) |
| L8 | Sin columnas 100 % nulas en el dataset final |

#### Dataset Final

```
master_v5_final.parquet
  101.976 filas × 215 columnas
  Periodo: 1997-08-17 → 2025-04-02
  Carreras: carrera_id 92 → 11926
  Carreras sintéticas: 0
```

> Las filas anteriores a 2005 se conservan como período de calentamiento (*warm-up*) para las ventanas históricas, pero no se usan como etiquetas de entrenamiento supervisado.

---

### `01_eda_calidad_fiabilidad.ipynb` — EDA, Auditoría y Preprocesamiento

Este cuaderno valida estadísticamente el dataset generado en el paso anterior y produce los artefactos de preprocesamiento que consume el notebook de modelado.

#### Decisiones metodológicas

| Decisión | Justificación |
|---|---|
| Excluir `carrera_id > 11926` | Carreras sin posición final registrada, no utilizables para entrenamiento supervisado |
| Excluir fechas anteriores a 1990 | Fechas anómalas que rompen la monotonía de `carrera_id` (Spearman ρ cae de 0,86 a 0,58) |
| Split por fecha, no por `carrera_id` | El identificador no es estrictamente monotónico; un split por ID filtraría carreras del siglo XIX en val/test |
| Split cronológico 70/20/10 | Refleja el caso de uso real: predecir carreras futuras con datos del pasado |
| Variables post-carrera excluidas | `pos`, `distancia_al_anterior`, `target_*`, `race_has_winner`; `odds_win` solo en baseline de mercado |

#### Secciones del análisis

El notebook recorre once secciones de análisis en orden:

**1. Auditoría de integridad temporal.** Verifica que `carrera_id` sea monotónico respecto a la fecha y detecta anomalías temporales. Descarta registros sin posición final registrada.

**2. Construcción del dataset limpio y splits temporales.** Aplica los filtros canónicos y genera los tres conjuntos de datos (train / val / test) mediante split cronológico por carrera.

**3. Auditoría de leakage.** Calcula el AUC univariante de cada variable frente al target. Variables con AUC > 0,70 se marcan como sospechosas. Las variables post-carrera identificadas se excluyen explícitamente del feature set.

**4. Análisis exploratorio univariante.** Distribuciones de las variables objetivo (`target_top1`, `target_top3`), evolución temporal del número de carreras y participaciones, y distribución por hipódromo y superficie.

**5. Calidad de features.** Porcentaje de nulos por variable, varianza cero y cobertura de entidades críticas (caballo, jinete, preparador).

**6. Covariate shift entre splits.** Test de Kolmogorov-Smirnov bilateral (H₀: misma distribución en train y test) para detectar cambios de distribución que puedan afectar a la generalización del modelo.

**7. Poder predictivo univariante.** AUC individual de cada variable frente a `target_top3`. Identifica las variables con mayor señal predictiva antes del modelado.

**8. Estabilidad temporal del poder predictivo.** Verifica que el AUC univariante de las variables más informativas se mantiene estable a lo largo del tiempo.

**9. Distribuciones, monotonía y correlaciones.** Matriz de correlaciones entre familias de variables y análisis de monotonía de las relaciones con el target.

**10. Preprocesamiento reproducible.** Ajusta e imputa valores nulos (mediana), estandariza (StandardScaler) y codifica variables categóricas (LabelEncoder). Todos los transformadores se ajustan exclusivamente sobre el conjunto de entrenamiento.

**11. Resumen ejecutivo y checks finales.** Verifica la coherencia global del dataset antes de pasar al modelado.

#### Artefactos persistidos

| Fichero | Contenido |
|---|---|
| `models/imputer.pkl` | `SimpleImputer(strategy='median')` ajustado sobre train |
| `models/scaler.pkl` | `StandardScaler` ajustado sobre train imputado |
| `models/label_encoders.pkl` | `LabelEncoder` por variable categórica |
| `models/feature_meta.json` | Lista de features, tipos, flags de leakage y columnas excluidas |

---

### `02_modelado_cientifico.ipynb` — Modelado y Evaluación

Este notebook entrena y compara todos los modelos sobre el mismo split temporal, calcula las métricas de ranking por carrera y analiza la interpretabilidad del mejor modelo.

#### Split temporal definitivo

| Partición | Periodo | Filas (%) | Carreras (%) |
|---|---|---|---|
| **Train** | 1997-01-19 → 2016-05-29 | 70 % | 70 % |
| **Validación** | 2016-06-03 → 2021-12-28 | 20 % | 20 % |
| **Test** | 2024-10-01 → 2025-04-02 | 10 % | 10 % |

El split se realiza por carrera completa (no por fila individual), garantizando que todos los participantes de una misma carrera caen en la misma partición.

#### Modelos evaluados

| Familia | Modelo | Configuración |
|---|---|---|
| Baselines | Bayes (tasa global), Casilla (posición de salida) | Sanity checks sin aprendizaje |
| Tree-based clasificadores | Random Forest, LightGBM, XGBoost, CatBoost | `early_stopping` sobre validación |
| Tree-based rankers | LGBMRanker, XGBRanker | Objetivo LambdaRank, grupos por carrera |
| Deep tabular | TabM (PLE + k-Ensemble), TabTransformer (embeddings + Transformer), MLP | PyTorch |
| HPO | **LightGBM_HPO** | Optuna TPE, 30 trials, objetivo NDCG@3 en validación |

> **LightGBM_HPO** es la variante de LightGBM con hiperparámetros optimizados mediante Optuna. El espacio de búsqueda incluye `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` y `scale_pos_weight`.

#### Métricas de evaluación

Todas las métricas se calculan por carrera y se agregan como media sobre el conjunto de test. Los intervalos de confianza al 95 % se estiman mediante bootstrap (B = 1000).

| Métrica | Tipo | Descripción |
|---|---|---|
| **NDCG@3** | Ranking | Normalized Discounted Cumulative Gain en los tres primeros puestos. Métrica principal. |
| **MAP@3** | Ranking | Mean Average Precision en los tres primeros puestos |
| **P@3** | Ranking | Precisión en los tres primeros puestos |
| **HR@1 / HR@3** | Ranking | Hit Rate: ¿aparece al menos un caballo del top-3 real entre los predichos? |
| **Spearman ρ** | Ranking | Correlación de Spearman entre ranking predicho y real |
| **AUC** | Clasificación | Área bajo la curva ROC (target_top3) |
| **F1** | Clasificación | F1-score binario |
| **Brier** | Calibración | Error cuadrático medio de las probabilidades predichas |
| **ECE** | Calibración | Expected Calibration Error (reliability diagram) |

La comparación entre modelos se realiza mediante el **test de Wilcoxon signed-rank pareado** sobre NDCG@3 por carrera (p < 0,05 para diferencia significativa).

#### Resultado principal

El mejor modelo en el conjunto de test es **LightGBM_HPO** con **NDCG@3 = 0,6055**, seguido de TabM como segunda mejor arquitectura.

#### Secciones del notebook

**1. Carga de datos y artefactos.** Lee el dataset final desde BigQuery (o parquet local) y carga los artefactos de preprocesamiento del notebook 01.

**2. Métricas de evaluación con inferencia estadística.** Define las funciones `race_metrics()`, `bootstrap_ci()` y `full_eval()` que se usan en todas las secciones de modelado.

**3. Baselines de referencia.** Evalúa la predicción aleatoria uniforme y la heurística de posición de salida (casilla). Establecen el umbral mínimo de rendimiento.

**4. Tree-based Classifiers.** Entrena Random Forest, LightGBM, XGBoost y CatBoost con `early_stopping` sobre validación. Incluye una ablación de Random Forest sin las variables `wr_distancia_m` (excluidas por ser constantes dentro de carrera).

**5. Rankers (LambdaRank).** Entrena LGBMRanker y XGBRanker con objetivo de ranking directo, usando los grupos de carrera como unidad de optimización.

**6. TabM — Piecewise Linear Encoding + k-Ensemble.** Implementación en PyTorch de la arquitectura TabM (Gorishniy et al., 2022). Codifica las variables numéricas mediante PLE sobre cuantiles del train y usa K cabezas independientes como ensemble interno.

**7. TabTransformer — Embeddings + Transformer Encoder.** Implementación en PyTorch de TabTransformer (Huang et al., 2020). Las variables categóricas se proyectan a embeddings aprendidos que atraviesan un Transformer Encoder; las numéricas se concatenan al output y pasan por un MLP final.

**8. HPO con Optuna (LightGBM, objetivo NDCG@3).** Optimización bayesiana con el algoritmo TPE de Optuna. Se ejecutan 30 trials maximizando NDCG@3 en validación. El mejor conjunto de hiperparámetros se usa para entrenar el modelo definitivo `LightGBM_HPO`.

**9. Tests estadísticos de comparación entre modelos.** Test de Wilcoxon signed-rank pareado sobre NDCG@3 por carrera para todos los pares de modelos. Identifica qué diferencias de rendimiento son estadísticamente significativas.

**10. Calibración.** Reliability diagrams y ECE para todos los modelos. Evalúa si las probabilidades predichas son fiables como estimaciones de probabilidad real.

**11. SHAP — Interpretabilidad del mejor modelo.** Valores SHAP mediante `TreeExplainer` sobre LightGBM_HPO. Identifica las 20 variables más influyentes y su dirección de efecto. Complementado con importancia por permutación sobre NDCG@3.

**12. Resumen final y persistencia.** Genera la tabla oficial de resultados (`model_results_test_official.csv`), guarda las predicciones del test (`test_predictions_official.npz`) y escribe el manifiesto de la ejecución en JSON.

---

## Requisitos

### Entorno recomendado

El código está diseñado para ejecutarse en **Google Colab Enterprise** o **Vertex AI Workbench** con acceso a Google Cloud Platform. Los notebooks incluyen un mecanismo de *fallback* que lee los datos desde disco local si las credenciales de GCP no están disponibles.

### Dependencias principales

```
pandas
numpy
scikit-learn
lightgbm
xgboost
catboost
torch
optuna
shap
scipy
matplotlib
seaborn
google-cloud-bigquery
google-cloud-storage
```

### Orden de ejecución

Los notebooks deben ejecutarse en orden secuencial:

```
00_pipeline_fuentes.ipynb
        ↓ genera master_v5_final.parquet
01_eda_calidad_fiabilidad.ipynb
        ↓ genera imputer.pkl, scaler.pkl, label_encoders.pkl, feature_meta.json
02_modelado_cientifico.ipynb
        ↓ genera model_results_test_official.csv, test_predictions_official.npz
```

Si ya se dispone del parquet procesado y los artefactos de preprocesamiento, puede ejecutarse directamente el notebook 02.

---

## Datos

Los datos de la Real Sociedad Hípica Española no se incluyen en este repositorio por razones de licencia.

Para reproducir el pipeline desde los CSVs raw es necesario disponer de acceso al proyecto GCP `project-bucephalus` o contar con los ficheros fuente en el directorio `data/raw/`.

---

## Referencia

Este repositorio acompaña al TFM *"Predicción de Resultados en Carreras de Caballos mediante Técnicas de Ranking Tabular"*, presentado en el Máster en Ingeniería de Inteligencia Artificial de la Universidad Internacional de La Rioja (UNIR), 2025-2026.
