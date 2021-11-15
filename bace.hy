(import os)
(import time)
(import [skopt [gp-minimize forest-minimize gbrt-minimize dummy-minimize]])
(import [skopt [dump load]])
(import [skopt.space :as space])
(import [skopt.utils [use-named-args]])
(import [skopt.plots [plot-convergence plot-objective plot-evaluations plot-regret]])
(import [numpy :as np])
(import [pandas :as pd])
(import [hace :as ac])
(import [fractions [Fraction]])
(import [functools [partial]])
(import [datetime [datetime :as dt]])
(import [matplotlib [pyplot :as plt]])
(require [hy.contrib.walk [let]])
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(import [hy.contrib.pprint [pp pprint]])

(setv ckt-path f"../ACE/ace/resource/xh035-3V3/st1"
      pdk-path  f"/mnt/data/pdk/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos"
      env (ac.schmitt-trigger ckt-path :pdk-path [pdk-path])
      design-parameters ["Wp0" "Wn0" "Wp2" "Wp1" "Wn2" "Wn1"]
      ;performance-parameters ["vs0" "vs1" "vs2" "vs3"]
      performance-parameters ["t_phl" "t_plh" "v_il" "v_ih"]
      ;performance-parameters ["v_il" "v_ih"]
      ;target (/ (get (ac.current-parameters env) "vdd") 2.0)
      w-min (np.array (list (repeat 0.4e-6 6)))
      w-max (np.array (list (repeat 150e-6 6))))

(defn unscale-value ^float [^float x′ ^float x-min ^float x-max
                  &optional ^float [a -1.0] ^float [b 1.0]]
  (+ x-min (* (/ (- x′ a) (- b a)) (- x-max x-min))))

(defn trigger [env X]
  (let [p design-parameters
        x (np.power 10 (- X))
        sizing (dict (zip p x))]
    (ac.evaluate-circuit env :params sizing)))

;(defn nand4 ^np.array [env ^np.array X]
;  (let [p ["wn0" "wn1" "wn2" "wn3" "wp"]
;        x (np.power 10 (- X))
;        sizing (dict (zip p x))]
;  (-> env (ac.evaluate-circuit :params sizing)
;          (np.array) (np.nan-to-num))))

(setv design-space (lfor p design-parameters
        (space.Real (np.abs (np.log10 150e-6)) 
                    (np.abs (np.log10 0.4e-6)) 
                    :name p :prior "uniform")))

;(setv targets (dict (zip performance-parameters (repeat target 4))))
(setv targets {"v_il" (- (/ 3.3 2.0) 0.3)
               "v_ih" (+ (/ 3.3 2.0) 0.3)
               "t_phl" 8e-10
               "t_plh" 8e-10})

(setv optimizer-functions [forest-minimize gbrt-minimize gp-minimize]
      base-estimators ["GP" "RF" "ET" "GBRT"]
      point-generators ["random" "sobol" "halton" "hammersly" "lhs" "grid"]
      acquisition-functions ["LCB" "EI" "PI" "gp_hedge"]  ;"EIps" "PIps"]
      acquisition-optimizer ["auto" "sampling" "lbfgs"]
      #_/ )

(setv sweep (lfor optimizer optimizer-functions 
                  estimator base-estimators 
                  generator point-generators 
                  acquisitor acquisition-functions
                (, optimizer estimator generator acquisitor)))

(setv t-getter (fn [d] (dfor k (.keys targets) [k (get d k)])))

(with-decorator (use-named-args :dimensions design-space)
  (defn design-objective [Wp0 Wn0 Wp2 Wp1 Wn2 Wn1]
    (let [x (np.array [Wp0 Wn0 Wp2 Wp1 Wn2 Wn1])
          y targets
          prf (->> x (trigger env) (t-getter) (.values) (list) (np.array))
          tgt (-> y       (t-getter) (.values) (list) (np.array))
          cost (/ (np.abs (- prf tgt)) tgt) ]
      (-> cost (np.nan-to-num) (np.sum)))))

;; Single Run Test
;(setv res (gp-minimize :func                    design-objective
;                       :dimensions              design-space 
;                       :n-calls                 100
;                       :n-random-starts         35
;                       :base-estimator          "GP"
;                       :initial-point-generator "halton"
;                       :acq-func                "PI"
;                       :xi                      0.01
;                       :kappa                   1.96
;                       :random-state            666
;                       :n-jobs                  42
;                       :verbose                 True))
;
;(plot-convergence res) (plt.show) 
;(plot-objective res) (plt.show) 
;(plot-objective res) (plt.savefig "./results/forest_minimize-ET-sobol-LCB.svg") 
;(plot-evaluations res) (plt.show) 

(for [s sweep]
  (let [(, optimizer estimator generator acquisitor) s
        time-stamp    (-> dt (.now) (.strftime "%H%M%S-%y%m%d"))
        run-name      (.format "{}-{}-{}-{}" optimizer.__name__ 
                               estimator generator acquisitor)
        model-prefix  (.format "./results/models/st1/{}-{}" run-name time-stamp)
        plot-prefix   (.format "./results/plots/st1/{}-{}" run-name time-stamp)
        res-path      (.format "{}-modl.pkl" model-prefix)
        res-conv-path (.format "{}-conv.svg" plot-prefix)
        res-objk-path (.format "{}-objk.svg" plot-prefix)
        res-eval-path (.format "{}-eval.svg" plot-prefix)
        _             (print (.format "{} | Starting {}" 
                                      (-> dt (.now) 
                                            (.strftime "%y-%m-%d - %H:%M:%S")) 
                                      run-name))
        t0 (time.process-time)
        res (optimizer :func                    design-objective
                       :dimensions              design-space 
                       :n-calls                 128
                       :n-random-starts         32
                       :base-estimator          estimator
                       :initial-point-generator generator
                       :acq-func                acquisitor
                       :xi                      0.01
                       :kappa                   1.96
                       :random-state            666
                       :n-jobs                  42
                       :verbose                 True)
        t1 (time.process-time)
        _ (print (.format "                    | Took {:.4f}s" (- t1 t0))) ]
    (print "                    | Dumping results")
    (dump res res-path)
    (print "                    | Plotting Convergence")
    (plot-convergence res) (plt.savefig res-conv-path) 
    (plt.close) (plt.cla) (plt.clf)
    (print "                    | Plotting Objective")
    (plot-objective res) (plt.savefig res-objk-path)
    (plt.close) (plt.cla) (plt.clf)
    (print "                    | Plotting Evaluations")
    (plot-evaluations res) (plt.savefig res-eval-path)
    (plt.close) (plt.cla) (plt.clf)
    (print "")
    ;(setv (get results run-name) res)
    #_/ ))

;;;;;;;;;;;;;;;;;;;;;;;;; LOAD RESULTS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(setv results 
  (dfor f (os.listdir "./results/models/st1") 
        :if         (.endswith f ".pkl")
        :setv k     (first (.split f "."))
        :setv (, o e g a) (take 4 (.split k "-"))
        :setv id    (.join "-" [o e g a])
        :setv ts    (.join "-" (take 2 (drop 4 (.split k "-"))))
        :setv res   (load f"./results/models/st1/{f}")
        :setv cost  (.item res.fun)
        :setv fmin  (-> res (. x) (np.array))
        :setv perf  (->> fmin (trigger env) (t-getter) (.values) (list))
        :setv desn  (.tolist (np.power 10 (- fmin)))
        :setv row   (+ [o e g a] desn perf [cost])
        :do         (print f"Loading {id}")
          [id {"res" res
               "ts" ts
               "row" row}]))

(setv column-names (+ ["optimizer" "estimator" "generator" "acquisitor"] 
                       design-parameters performance-parameters 
                       ["cost" "time_stamp"])
      df (pd.DataFrame
            (lfor v (.values results) (+ (get v "row") [(get v "ts")]))
            :columns column-names))


(setv (get df "time_stamp") 
(lfor ts df.time-stamp.values (.join "-" (take 2 (.split ts "-")))))


(df.to-csv "./results/st1.csv" :index False)

(setv best-key (first (list (filter #%(.startswith %1 "gp_minimize-GBRT-sobol-PI") 
                                      (.keys results)))))

(setv best-model (get results best-key))

(setx md (.join f"\n" 
(lfor i (range (first df.shape))
  :setv row (get df.loc i)
  :setv id (.join "-" (.tolist (. (get row ["optimizer" "estimator" 
                                            "generator" "acquisitor"]) 
                                  values)))
  :setv res (get results id)
  :setv ts (.join "-" (take 2 (.split (get res "ts") "-"))) ;(get res "ts")
  :setv plot f"./plots/st1/{id}-{ts}-objk.svg"
f"
### {id}
```
{(.to-string row)}
```\n
![{id}]({plot})
<p align=\"center\">{id}</p>\n
")))

(with [f (open "./results/st1.md" "w")]
  (f.write 
f"# Comparison\n
Detailed comparison of optimization algorithms.
")
  (f.write md))




(setv markdown 
      (lfor (, k v) (.items results)
            (setv (, o e g a) (-> v (get "plt") (.split "-") (take 4))
                  loc (get df (and (= df.optimizer o)
                                 (= df.estimator e)
                                 (= df.estimator e)
                              ))
                  fval (-> v (get "res") (.fun))
                  fmin (np.power 10 (-> v (get "res") (. x) (np.array) (-)))
                  objk (get v "plt"))
f"<h3 align=\"center\">{o}</h3>
![]({objk})
|-------------+--------|
| Optimizer   | {o}    |
| Estimator   | {e}    |
| Generator   | {g}    |
| Acquisition | {a}    |
| Minimum     | {fval} |
| Parameters  | {fmin} |
|-------------+--------|
"))



