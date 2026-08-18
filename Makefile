PYTHON ?= /opt/miniconda3/bin/python3
DATA   ?= PartdData
DATA_A ?= PartaData
DATA_B ?= PartbData
OUT    ?= out

TRAIN_DIR_d1 := $(DATA)/random_train
TEST_DIR_d1  := $(DATA)/random_test
TRAIN_DIR_d2 := $(DATA)/train_set
TEST_DIR_d2  := $(DATA)/test_set
TRAIN_DIR_d3 := $(DATA)/train_d3
TEST_DIR_d3  := $(DATA)/test_d3

D3_TRAIN_SUBJECTS := c1s01 c1s02 c1s03 c1s05 c2s01 c2s02 c2s04 c2s05
D3_TEST_SUBJECTS  := c1s04 c2s03

.DEFAULT_GOAL := help
.PHONY: all d1 d2 d3 d3-data clean help \
        diagnose-d1 diagnose-d2 diagnose-d3 \
        inspect-d1 inspect-d2 inspect-d3 \
        compare-d1 compare-d2 compare-d3 \
        eval-a eval-b eval-c eval-d1 eval-d2 eval-d3 eval

help:
	@echo "make d1 | d2 | d3               train + build test features for one protocol"
	@echo "make all                        do all three protocols"
	@echo "make diagnose-d1 | -d2 | -d3    train with correlation printing + NMAE/NMSE vs. median baseline"
	@echo "                                (dev-only: needs the dev test dir's glucose field)"
	@echo "make compare-d1 | -d2 | -d3     compare Ridge vs. Lasso vs. ElasticNet on identical features"
	@echo "                                (dev-only: needs the dev test dir's glucose field)"
	@echo "make inspect-d1 | -d2 | -d3     print trained model's per-feature medians"
	@echo "make d3-data                    (re)build the train_d3/test_d3 symlink dirs from train_set/test_set"
	@echo "make eval-a                     run the official eval/eval_a.py evaluator on part_a.py"
	@echo "make eval-b                     run the official eval/eval_b.py evaluator on part_b.py"
	@echo "make eval-c                     run the official eval/eval_c.py evaluator on part_c.py"
	@echo "                                (part_c.py doesn't exist in this repo yet)"
	@echo "make eval-d1 | -d2 | -d3        run the official eval/eval_d.py evaluator (builds model+features first)"
	@echo "make eval                       eval-a + eval-d1 + eval-d2 + eval-d3 (eval-b/-c excluded: see above)"
	@echo "make clean                      remove generated out/ (models + feature files)"
	@echo ""
	@echo "override on the command line if needed, e.g.:"
	@echo "  make PYTHON=python3 d1"
	@echo "  make DATA=/path/to/data d2"

all: d1 d2 d3

# make d1 / make d2 / make d3 -> train, then build test features, for that protocol
d1 d2 d3: %: $(OUT)/model_%.pkl $(OUT)/features_%.npy

$(OUT):
	mkdir -p $(OUT)

$(OUT)/model_%.pkl: part_d.py | $(OUT)
	$(PYTHON) part_d.py train $* $(TRAIN_DIR_$*) $@

$(OUT)/features_%.npy: $(OUT)/model_%.pkl part_d.py | $(OUT)
	$(PYTHON) part_d.py feature_engineering $* $(TEST_DIR_$*) $(OUT)/model_$*.pkl $@

# Dev-only diagnostics (not the graded train/feature_engineering interface).
# diagnose-% always re-runs (PHONY): trains fresh with correlation printing
# on, then reports NMAE/NMSE vs. a median-training-target baseline using the
# *dev* test dir's glucose field. Point TEST_DIR_% at a real held-out test
# set (no glucose) and this will fail -- that's intentional.
diagnose-d1 diagnose-d2 diagnose-d3: diagnose-%: part_d.py | $(OUT)
	$(PYTHON) part_d.py diagnose eval $* $(TRAIN_DIR_$*) $(TEST_DIR_$*) $(OUT)/model_$*.pkl

# compare-%: same dev-test-glucose requirement as diagnose-%, but fits
# Ridge/Lasso/ElasticNet on identical features and reports all three --
# saves no model file, purely a comparison.
compare-d1 compare-d2 compare-d3: compare-%: part_d.py
	$(PYTHON) part_d.py diagnose compare $* $(TRAIN_DIR_$*) $(TEST_DIR_$*)

# inspect-% reuses the normal model_%.pkl target -- builds it first if
# missing/stale, otherwise just inspects what's already there.
inspect-d1 inspect-d2 inspect-d3: inspect-%: $(OUT)/model_%.pkl
	$(PYTHON) part_d.py diagnose inspect $(OUT)/model_$*.pkl

# d3's train/test dirs aren't shipped directly -- train_set/test_set are laid
# out for d2 (same participant, earlier vs. later segment). This rebuilds the
# disjoint-subject d3 layout via symlinks, per PartdData/Data_organization.md.
# Already present in this repo, but re-run if PartdData is ever re-extracted fresh.
d3-data:
	mkdir -p $(DATA)/train_d3 $(DATA)/test_d3
	for s in $(D3_TRAIN_SUBJECTS); do \
		ln -sf ../train_set/$${s}_a.npz $(DATA)/train_d3/$${s}_a.npz; \
		ln -sf ../test_set/$${s}_b.npz  $(DATA)/train_d3/$${s}_b.npz; \
	done
	for s in $(D3_TEST_SUBJECTS); do \
		ln -sf ../train_set/$${s}_a.npz $(DATA)/test_d3/$${s}_a.npz; \
		ln -sf ../test_set/$${s}_b.npz  $(DATA)/test_d3/$${s}_b.npz; \
	done
	@echo "train_d3: $$(ls $(DATA)/train_d3 | wc -l) files (expect 16)"
	@echo "test_d3:  $$(ls $(DATA)/test_d3 | wc -l) files (expect 4)"

# Official evaluator scripts (eval/eval_*.py) -- these are the graders'
# own checkers, separate from the dev-only diagnose-%/compare-%/inspect-%
# targets above.
#
# All eval-* targets below wrap the eval invocation with a wall-clock timer
# so you can read off actual elapsed seconds -- eval_a.py/eval_d.py already
# enforce the PDF's time limit internally (eval_a.py fails the subprocess
# past --timeout) but only report pass/fail, not how much margin you had.
eval-a: part_a.py
	@start=$$(date +%s); \
	$(PYTHON) eval/eval_a.py part_a.py $(DATA_A)/e4_hr_train_downsampled.csv $(DATA_A)/e4_hr_test_downsampled.csv; \
	status=$$?; \
	echo "[eval-a] wall time: $$(($$(date +%s) - start))s"; \
	exit $$status

# eval-b: PartbData/ holds folds.txt + regularization.txt, with the
# train/test CSVs symlinked in from PartaData/ (part (b) uses the same
# 1640-feature/hr schema and the PDF explicitly says to reuse part (a)'s
# feature representation, so no need to duplicate the ~1.5GB of CSVs).
eval-b: part_b.py
	@start=$$(date +%s); \
	$(PYTHON) eval/eval_b.py part_b.py $(DATA_B)/e4_hr_train_downsampled.csv $(DATA_B)/e4_hr_test_downsampled.csv $(DATA_B)/folds.txt $(DATA_B)/regularization.txt; \
	status=$$?; \
	echo "[eval-b] wall time: $$(($$(date +%s) - start))s"; \
	exit $$status

# eval-c: same train/test CSVs as (a)/(b) per the PDF. part_c.py doesn't
# exist in this repo yet -- this target will fail with "No rule to make
# target 'part_c.py'" until it's added.
eval-c: part_c.py
	@start=$$(date +%s); \
	$(PYTHON) eval/eval_c.py part_c.py $(DATA_A)/e4_hr_train_downsampled.csv $(DATA_A)/e4_hr_test_downsampled.csv; \
	status=$$?; \
	echo "[eval-c] wall time: $$(($$(date +%s) - start))s"; \
	exit $$status

# eval-d%: always reruns train + feature_engineering from scratch (PHONY,
# like diagnose-%/compare-% -- deliberately ignores any cached
# model_%.pkl/features_%.npy from a previous `make d%`) so the timing below
# is a real, fresh measurement against the PDF's 30-min-per-protocol
# train+feature_engineering limit, then runs the official eval_d.py check
# against the labelled test dir.
eval-d1 eval-d2 eval-d3: eval-%: part_d.py | $(OUT)
	@start=$$(date +%s); \
	$(PYTHON) part_d.py train $* $(TRAIN_DIR_$*) $(OUT)/model_$*.pkl && \
	$(PYTHON) part_d.py feature_engineering $* $(TEST_DIR_$*) $(OUT)/model_$*.pkl $(OUT)/features_$*.npy; \
	status=$$?; \
	mid=$$(date +%s); \
	echo "[eval-$*] train+feature_engineering wall time: $$((mid-start))s (PDF limit: 1800s)"; \
	if [ $$status -eq 0 ]; then \
		$(PYTHON) eval/eval_d.py --model $(OUT)/model_$*.pkl --features $(OUT)/features_$*.npy --labels $(TEST_DIR_$*); \
		status=$$?; \
	fi; \
	echo "[eval-$*] total wall time incl. eval check: $$(($$(date +%s) - start))s"; \
	exit $$status

eval: eval-a eval-d1 eval-d2 eval-d3

clean:
	rm -rf $(OUT)
