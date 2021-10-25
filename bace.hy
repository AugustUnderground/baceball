;(import ray)
(import time)
(import [skopt [gp-minimize forest-minimize gbrt-minimize dummy-minimize]])
(import [skopt [dump load]])
(import [skopt.space :as space])
(import [skopt.utils [use-named-args]])
(import [skopt.plots [plot-convergence plot-objective plot-evaluations]])
(import [numpy :as np])
(import [pandas :as pd])
(import [torch :as pt])
(import [hace :as ac])
(import [joblib :as jl])
(import [fractions [Fraction]])
(import [functools [partial]])
(import [datetime [datetime :as dt]])
(import [matplotlib [pyplot :as plt]])
(require [hy.contrib.walk [let]])
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(import [hy.contrib.pprint [pp pprint]])

(defclass PrimitiveDevice []
  (defn __init__ [self ^str model-path ^str scale-x-path ^str scale-y-path]
    (setv self.path     model-path
          self.params-x ["gmoverid" "fug" "Vds" "Vbs"]
          self.params-y ["idoverw" "L" "gdsoverw" "Vgs"]
          self.trafo-x  ["fug"]
          self.trafo-y  ["idoverw" "gdsoverw"]
          self.mask-x   (np.array (lfor px self.params-x (int (in px self.trafo-x))))
          self.mask-y   (np.array (lfor py self.params-y (int (in py self.trafo-y)))))
    (setv self.scaler-x (jl.load scale-x-path)
          self.scaler-y (jl.load scale-y-path)
          self.scale-x  (fn [X] (self.scaler-x.transform X))
          self.scale-y  (fn [Y] (self.scaler-y.inverse-transform Y)))
    (setv self.trafo-x  (fn [X] (+ (* (np.log10 (np.abs X) 
                                                :where (> (np.abs X) 0)) 
                                      self.mask-x) 
                                   (* X (- 1 self.mask-x))))
          self.trafo-y  (fn [Y] (+ (* (np.power 10 Y) self.mask-y) 
                                   (* Y (- 1 self.mask-y)))))
    (setv self.model (-> self.path (pt.jit.load) (.cpu) (.eval))))
  (defn predict ^np.array [self ^np.array X]
    (with [_ (pt.no-grad)]
      (-> X (self.trafo-x) 
            (self.scale-x) 
            (np.float32) 
            (pt.from-numpy) 
            (self.model) 
            (.numpy) 
            (self.scale-y) 
            (self.trafo-y)))))

(setv target-values
  {"a_0"          105.0
   "ugbw"         3500000.0
   "pm"           110.0
   "gm"           -45.0
   "sr_r"         2700000.0
   "sr_f"         -2700000.0
   "vn_1Hz"       6.0e-06
   "vn_10Hz"      2.0e-06
   "vn_100Hz"     6.0e-07
   "vn_1kHz"      1.5e-07
   "vn_10kHz"     5.0e-08
   "vn_100kHz"    2.6e-08
   "psrr_n"       120.0
   "psrr_p"       120.0
   "cmrr"         110.0
   "v_il"         0.7
   "v_ih"         3.2
   "v_ol"         0.1
   "v_oh"         3.2
   "i_out_min"    -7e-5
   "i_out_max"    7e-5
   "overshoot_r"  0.0005
   "overshoot_f"  0.0005
   "voff_stat"    3e-3
   "voff_sys"     -2.5e-05
   #_/ })

;(setv target-values
;  {"a_0"         55.0
;   "ugbw"        3750000.0
;   "pm"          65.0
;   "gm"          -30.0
;   "sr_r"        3750000.0
;   "sr_f"        -3750000.0
;   "vn_1Hz"      5e-06
;   "vn_10Hz"     2e-06
;   "vn_100Hz"    5e-07
;   "vn_1kHz"     1.5e-07
;   "vn_10kHz"    5e-08
;   "vn_100kHz"   2.5e-08
;   "psrr_n"      80.0
;   "psrr_p"      80.0
;   "cmrr"        80.0
;   "v_il"        0.9
;   "v_ih"        3.2
;   "v_ol"        1.65
;   "v_oh"        3.2
;   "i_out_min"   -7e-5
;   "i_out_max"   7e-5
;   "overshoot_r" 2.0
;   "overshoot_f" 2.0
;   "voff_stat"   3e-3
;   "voff_sys"    -1.5e-3
;   #_/ })

(setv target-predicates
  {"a_0"         '<=
   "ugbw"        '<=
   "pm"          '<=
   "gm"          '>=
   "sr_r"        '<=
   "sr_f"        '>=
   "vn_1Hz"      '>=
   "vn_10Hz"     '>=
   "vn_100Hz"    '>=
   "vn_1kHz"     '>=
   "vn_10kHz"    '>=
   "vn_100kHz"   '>=
   "psrr_n"      '<=
   "psrr_p"      '<=
   "cmrr"        '<=
   "v_il"        '>=
   "v_ih"        '<=
   "v_ol"        '>=
   "v_oh"        '<=
   "i_out_min"   '>=
   "i_out_max"   '>=
   "overshoot_r" '>=
   "overshoot_f" '>=
   "voff_stat"   '>=
   "voff_sys"    '<=
   #_/ })

(setv model-base-path f"../models/xh035"
      nmos (PrimitiveDevice f"{model-base-path}-nmos/model.pt"
                            f"{model-base-path}-nmos/scale.X"
                            f"{model-base-path}-nmos/scale.Y")
      pmos (PrimitiveDevice f"{model-base-path}-pmos/model.pt"
                            f"{model-base-path}-pmos/scale.X"
                            f"{model-base-path}-pmos/scale.Y"))

;(defn op2 ^np.array [amp ^np.array x]
(defn op2 ^np.array [^np.array x]
  (let [ (, gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1
            fug-cm1  fug-cm2  fug-cm3  fug-dp1 
            i1 i2 ) x
          (, Mcm31 Mcm32 Mdp1) (, 2 2 2)
          vsup (-> amp (ac.current-parameters) (get "vsup"))
          i0   (-> amp (ac.current-parameters) (get "i0"))
          vx   0.2
          M1   (-> (/ i0 i1) (Fraction) (.limit-denominator 100))
          M2   (-> (/ (/ i1 2) i2) (Fraction) (.limit-denominator 100))
          (, Mcm11 Mcm12) (, M1.numerator M1.denominator)
          (, Mcm21 Mcm22) (, M2.numerator M2.denominator)
          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ vsup 2) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ vsup 2) 0.0]])
          cm3-in (np.array [[gmid-cm3 fug-cm3 (/ vsup 2) 0.0]])
          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ vsup 2) 0.0]])
          cm1-out (first (nmos.predict cm1-in))
          cm2-out (first (pmos.predict cm2-in))
          cm3-out (first (nmos.predict cm3-in))
          dp1-out (first (nmos.predict dp1-in))
          Lcm1 (get cm1-out 1)
          Lcm2 (get cm2-out 1)
          Lcm3 (get cm3-out 1)
          Ldp1 (get dp1-out 1)
          Wcm1 (/ i0 (get cm1-out 0))
          Wcm2 (/ (* 0.5 i1) (get cm2-out 0))
          Wcm3 (/ i2 (get cm3-out 0))
          Wdp1 (/ (* 0.5 i1) (get dp1-out 0)) 
          sizing { "Lcm1"  Lcm1  "Lcm2"  Lcm2  "Lcm3"  Lcm3  "Ld" Ldp1
                   "Wcm1"  Wcm1  "Wcm2"  Wcm2  "Wcm3"  Wcm3  "Wd" Wdp1
                   "Mcm11" Mcm11 "Mcm21" Mcm21 "Mcm31" Mcm31
                   "Mcm12" Mcm12 "Mcm22" Mcm22 "Mcm32" Mcm32
                   "Md"    Mdp1 } ]
  (-> amp (ac.evaluate-circuit :params sizing)
          (np.array) (np.nan-to-num))))

;(op2 (np.array [10.0 10.0 10.0 10.0 8.0 8.0 8.0 8.0 4.0e-6 50.0e-6]))
;(op2 (np.random.rand 10))

(setv design-space 
  [ (space.Real 7.0 17.0  :name "gmid_cm1" :prior "normal")
    (space.Real 7.0 17.0  :name "gmid_cm2" :prior "normal")
    (space.Real 7.0 17.0  :name "gmid_cm3" :prior "normal")
    (space.Real 7.0 17.0  :name "gmid_dp1" :prior "normal")
    (space.Real 1e6 1e9   :name "fug_cm1"  :prior "uniform")
    (space.Real 1e6 1e9   :name "fug_cm2"  :prior "uniform")
    (space.Real 1e6 1e9   :name "fug_cm3"  :prior "uniform")
    (space.Real 1e6 1e9   :name "fug_dp1"  :prior "uniform")
    (space.Real 3e-6 5e-5 :name "i1"       :prior "normal")
    (space.Real 3e-6 5e-4 :name "i2"       :prior "normal") ])

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

;(setv t-getter (itemgetter #* (.keys target-values)))
(setv t-getter (fn [d] (dfor k (.keys target-values) [k (get d k)]))
      ec ["MNCM11:gmoverid" "MPCM211:gmoverid" "MNCM31:gmoverid" "MND11:gmoverid"
          "MNCM11:fug" "MPCM211:fug" "MNCM31:fug" "MND11:fug" "MNCM11:id" "MNCM31:id"]
      e-getter (fn [d] (dfor k ec [k (get d k)])))

(setv ckt-path f"../ACE/ace/resource/xh035-3V3/op2"
      pdk-path  f"/mnt/data/pdk/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos"
      amp (ac.single-ended-opamp ckt-path :pdk-path [pdk-path]))

(with-decorator (use-named-args :dimensions design-space)
  (defn design-objective [gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1 
                          fug-cm1 fug-cm2 fug-cm3 fug-dp1 i1 i2]
    (let [x (np.array [gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1 
                       fug-cm1 fug-cm2 fug-cm3 fug-dp1 i1 i2])
          y target-values
          msk (-> target-predicates (t-getter) (.values) (list))
          prf (-> x (op2) (t-getter) (.values) (list) (np.array))
          tgt (-> y       (t-getter) (.values) (list) (np.array))
          loss (/ (np.abs (- prf tgt)) tgt)
          mask (lfor (, c p t) (zip msk prf tgt)
                     ((eval c) t p))
          cost (+ (* (np.tanh (np.abs loss)) mask) 
                  (* (- (** loss 2.0)) (np.invert mask))) ]
      (-> cost (np.nan-to-num) (np.sum) (-)))))

(setv t target-values
      p (ac.evaluate-circuit amp)
      d (dfor k (.keys target-values) 
  [k [ (get t k) (get p k)
      (/ (np.abs (- (get p k) (get t k))) (get t k))]] ))

(setv df (pd.DataFrame.from-dict d :orient "index" :columns ["t" "p" "d"]))

;; Single run
(setv res (gp-minimize :func                    design-objective
                       :dimensions              design-space 
                       :n-calls                 42
                       :n-random-starts         15
                       :base-estimator          "ET"
                       :initial-point-generator "sobol"
                       :acq-func                "LCB"
                       :xi                      0.01
                       :kappa                   1.96
                       :random-state            666
                       :n-jobs                  42
                       :verbose                 True))

(plot-convergence res) (plt.show) 
(plot-objective res) (plt.show) 
(plot-objective res) (plt.savefig "./results/forest_minimize-ET-sobol-LCB.svg") 
(plot-evaluations res) (plt.show) 







(for [s sweep]
  (let [(, optimizer estimator generator acquisitor) s
        time-stamp (-> dt (.now) (.strftime "%H%M%S-%y%m%d"))
        run-name (.format "{}-{}-{}-{}" optimizer.__name__ 
                          estimator generator acquisitor)
        res-prefix (.format "./results/{}-{}" run-name time-stamp)
        res-path (.format "{}.pkl" res-prefix)
        res-conv-path (.format "{}-conv.svg" res-prefix)
        res-objk-path (.format "{}-objk.svg" res-prefix)
        res-eval-path (.format "{}-eval.svg" res-prefix)
        _ (print (.format "{} | Starting {}" 
                          (-> dt (.now) (.strftime "%y-%m-%d - %H:%M:%S")) 
                          run-name))
        t0 (time.process-time)
        res (optimizer :func                    design-objective
                       :dimensions              design-space 
                       :n-calls                 5
                       :n-random-starts         2
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
    ))

(plot-convergence #* results) (plt.show)
(plot-convergence #* results) (plt.savefig "compare-all-cong.svg")











(with-decorator ray.remote
(defclass OPWorker []
  (defn __init__ [self tv tp ds]
    (setv self.sim-path "/tmp"
          self.pdk-path "/mnt/data/pdk/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos"
          self.ckt-path "../ACE/ace/resource/xh035-3V3/op2"
          self.target-values tv
          self.target-predicates tp
          self.design-space ds
          self.t-getter (fn [d] (dfor k (.keys tv) [k (get d k)]))
          self.op (ac.single-ended-opamp self.ckt-path 
                                  :pdk-path [self.pdk-path] 
                                  :sim-path self.sim-path)
          self.model-base-path f"../models/xh035"
          self.nmos (PrimitiveDevice.remote f"{self.model-base-path}-nmos/model.pt"
                            f"{self.model-base-path}-nmos/scale.X"
                            f"{self.model-base-path}-nmos/scale.Y")
          self.pmos (PrimitiveDevice.remote f"{self.model-base-path}-pmos/model.pt"
                            f"{self.model-base-path}-pmos/scale.X"
                            f"{self.model-base-path}-pmos/scale.Y")))
   (defn performance [self x]
     (let [ (, gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1
               fug-cm1  fug-cm2  fug-cm3  fug-dp1 
               i1 i2 ) x
             (, Mcm31 Mcm32 Mdp1) (, 2 2 2)
             vsup (-> self.op (ac.current-parameters) (get "vsup"))
             i0   (-> self.op (ac.current-parameters) (get "i0"))
             vx   0.2
             M1   (-> (/ i0 i1) (Fraction) (.limit-denominator 100))
             M2   (-> (/ (/ i1 2) i2) (Fraction) (.limit-denominator 100))
             (, Mcm11 Mcm12) (, M1.numerator M1.denominator)
             (, Mcm21 Mcm22) (, M2.numerator M2.denominator)
             cm1-in (np.array [[gmid-cm1 fug-cm1 (/ vsup 2) 0.0]])
             cm2-in (np.array [[gmid-cm2 fug-cm2 (/ vsup 2) 0.0]])
             cm3-in (np.array [[gmid-cm3 fug-cm3 (/ vsup 2) 0.0]])
             dp1-in (np.array [[gmid-dp1 fug-dp1 (/ vsup 2) 0.0]])
             cm1-out (first (ray.get (self.nmos.predict.remote cm1-in)))
             cm2-out (first (ray.get (self.pmos.predict.remote cm2-in)))
             cm3-out (first (ray.get (self.nmos.predict.remote cm3-in)))
             dp1-out (first (ray.get (self.nmos.predict.remote dp1-in)))
             Lcm1 (get cm1-out 1)
             Lcm2 (get cm2-out 1)
             Lcm3 (get cm3-out 1)
             Ldp1 (get dp1-out 1)
             Wcm1 (/ i0 (get cm1-out 0))
             Wcm2 (/ (* 0.5 i1) (get cm2-out 0))
             Wcm3 (/ i2 (get cm3-out 0))
             Wdp1 (/ (* 0.5 i1) (get dp1-out 0)) 
             sizing { "Lcm1"  Lcm1  "Lcm2"  Lcm2  "Lcm3"  Lcm3  "Ld" Ldp1
                      "Wcm1"  Wcm1  "Wcm2"  Wcm2  "Wcm3"  Wcm3  "Wd" Wdp1
                      "Mcm11" Mcm11 "Mcm21" Mcm21 "Mcm31" Mcm31
                      "Mcm12" Mcm12 "Mcm22" Mcm22 "Mcm32" Mcm32
                      "Md"    Mdp1 } ]
     (-> self.op (ac.evaluate-circuit :params sizing)
             (np.array) (np.nan-to-num))))
  (with-decorator (ray.method :num-returns 1)
  (defn optim-run [self optimizer estimator generator acquisitor]
    (let [ ;run-name (.format "{}-{}-{}-{}" optimizer.__name__ 
           ;                 estimator generator acquisitor)
          design-objective (with-decorator (use-named-args :dimensions design-space)
              (fn [gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1 
                   fug-cm1 fug-cm2 fug-cm3 fug-dp1 i1 i2]
                (let [x (np.array [gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1 
                                   fug-cm1 fug-cm2 fug-cm3 fug-dp1 i1 i2])
                      y self.target-values
                      msk (-> self.target-predicates (self.t-getter) (.values) (list))
                      prf (-> x (self.performance) (self.t-getter) (.values) (list) (np.array))
                      tgt (-> y       (self.t-getter) (.values) (list) (np.array))
                      loss (/ (np.abs (- prf tgt)) tgt)
                      mask (lfor (, c p t) (zip msk prf tgt)
                                 ((eval c) t p))
                      cost (+ (* (np.tanh (np.abs loss)) mask) 
                              (* (- (** loss 2.0)) (np.invert mask))) ]
                  (-> cost (np.nan-to-num) (np.sum) (-)))))
          res (optimizer :func                    design-objective
                         :dimensions              self.design-space 
                         :n-calls                 5
                         :n-random-starts         1
                         ;:base-estimator          estimator
                         :initial-point-generator generator
                         :acq-func                acquisitor
                         :xi                      0.01
                         :kappa                   1.96
                         :random-state            666
                         :n-jobs                  1
                         :verbose                 True)
          ]
      42)))
  (defn rng [self a b c d]
    (ac.random-sizing self.op))
  ))

(ray.init)

(setv worker (.remote OPWorker target-values target-predicates design-space))
(setv thunk (worker.optim-run.remote forest-minimize "RF" "sobol" "PI"))
(setv result (ray.get thunk))

(setv thunk (worker.performance.remote (np.random.rand 10)))
(setv result (ray.get thunk))

(setv workers (lfor _ (range 5) (.remote OPWorker target-values target-predicates design-space)))

(setv thunks (lfor w workers (.remote w.optim-run forest-minimize "RF" "sobol" "PI")))

(setv results (-> thunks (ray.get)))

(setv res (pd.concat (lfor _ (range 666) 
  (-> (lfor w workers (.remote w.sample)) 
      (ray.get) (pd.concat :ignore-index True))) :ignore-index True))


















;(defn op2 ^np.array [amp ^np.array x]
(defn op2 ^np.array [^np.array x]
  (let [ (, gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1
            fug-cm1  fug-cm2  fug-cm3  fug-dp1 
            i1 i2 ) x
          (, Mcm31 Mcm32 Mdp1) (, 2 2 2)
          vsup (-> amp (ac.current-parameters) (get "vsup"))
          i0   (-> amp (ac.current-parameters) (get "i0"))
          vx   0.2
          M1   (-> (/ i0 i1) (Fraction) (.limit-denominator 100))
          M2   (-> (/ (/ i1 2) i2) (Fraction) (.limit-denominator 100))
          (, Mcm11 Mcm12) (, M1.numerator M1.denominator)
          (, Mcm21 Mcm22) (, M2.numerator M2.denominator)
          cm1-in (np.array [[gmid-cm1 fug-cm1 (/ vsup 2) 0.0]])
          cm2-in (np.array [[gmid-cm2 fug-cm2 (/ vsup 2) 0.0]])
          cm3-in (np.array [[gmid-cm3 fug-cm3 (/ vsup 2) 0.0]])
          dp1-in (np.array [[gmid-dp1 fug-dp1 (/ vsup 2) 0.0]])
          cm1-out (first (nmos.predict cm1-in))
          cm2-out (first (pmos.predict cm2-in))
          cm3-out (first (nmos.predict cm3-in))
          dp1-out (first (nmos.predict dp1-in))
          Lcm1 (get cm1-out 1)
          Lcm2 (get cm2-out 1)
          Lcm3 (get cm3-out 1)
          Ldp1 (get dp1-out 1)
          Wcm1 (/ i0 (get cm1-out 0))
          Wcm2 (/ (* 0.5 i1) (get cm2-out 0))
          Wcm3 (/ i2 (get cm3-out 0))
          Wdp1 (/ (* 0.5 i1) (get dp1-out 0)) 
          sizing { "Lcm1"  Lcm1  "Lcm2"  Lcm2  "Lcm3"  Lcm3  "Ld" Ldp1
                   "Wcm1"  Wcm1  "Wcm2"  Wcm2  "Wcm3"  Wcm3  "Wd" Wdp1
                   "Mcm11" Mcm11 "Mcm21" Mcm21 "Mcm31" Mcm31
                   "Mcm12" Mcm12 "Mcm22" Mcm22 "Mcm32" Mcm32
                   "Md"    Mdp1 } ]
  (-> amp (ac.evaluate-circuit :params sizing)
          (np.array) (np.nan-to-num))))

;(op2 (np.array [10.0 10.0 10.0 10.0 8.0 8.0 8.0 8.0 4.0e-6 50.0e-6]))
;(op2 (np.random.rand 10))


(with-decorator (use-named-args :dimensions design-space)
  (defn design-objective [gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1 
                          fug-cm1 fug-cm2 fug-cm3 fug-dp1 i1 i2]
    (let [x (np.array [gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1 
                       fug-cm1 fug-cm2 fug-cm3 fug-dp1 i1 i2])
          y target-values
          msk (-> target-predicates (t-getter) (.values) (list))
          prf (-> x (op2) (t-getter) (.values) (list) (np.array))
          tgt (-> y       (t-getter) (.values) (list) (np.array))
          loss (/ (np.abs (- prf tgt)) tgt)
          mask (lfor (, c p t) (zip msk prf tgt)
                     ((eval c) t p))
          cost (+ (* (np.tanh (np.abs loss)) mask) 
                  (* (- (** loss 2.0)) (np.invert mask))) ]
      (-> cost (np.nan-to-num) (np.sum) (-)))))


;(setv model-base-path f"../models/xh035"
;      nmos (PrimitiveDevice f"{model-base-path}-nmos/model.pt"
;                            f"{model-base-path}-nmos/scale.X"
;                            f"{model-base-path}-nmos/scale.Y")
;      pmos (PrimitiveDevice f"{model-base-path}-pmos/model.pt"
;                            f"{model-base-path}-pmos/scale.X"
;                            f"{model-base-path}-pmos/scale.Y"))

;(setv t-getter (itemgetter #* (.keys target-values)))
;(setv t-getter (fn [d] (dfor k (.keys target-values) [k (get d k)])))

;(setv ckt-path f"../ACE/ace/resource/xh035-3V3/op2"
;      pdk-path  f"/mnt/data/pdk/XKIT/xh035/cadence/v6_6/spectre/v6_6_2/mos"
;      amp (ac.single-ended-opamp ckt-path :pdk-path [pdk-path])
;      ps (list (map str (filter #%(.islower %1) (ac.performance-identifiers amp))))
;      p-getter (itemgetter #* ps))









(setv results {})

;(lfor optimizer minimizer-functions 
;      estimator base-estimators 
;      generator point-generators 
;      acquisitor acquisition-functions

(for [s sweep]
  (let [(, optimizer estimator generator acquisitor) s
        time-stamp (-> dt (.now) (.strftime "%H%M%S-%y%m%d"))
        run-name (.format "{}-{}-{}-{}" optimizer.__name__ 
                          estimator generator acquisitor)
        res-prefix (.format "./results/{}-{}" run-name time-stamp)
        res-path (.format "{}.pkl" res-prefix)
        res-conv-path (.format "{}-conv.svg" res-prefix)
        res-objk-path (.format "{}-objk.svg" res-prefix)
        res-eval-path (.format "{}-eval.svg" res-prefix)
        _ (print (.format "{} | Starting {}" 
                          (-> dt (.now) (.strftime "%y-%m-%d - %H:%M:%S")) 
                          run-name))
        t0 (time.process-time)
        res (optimizer :func                    design-objective
                       :dimensions              design-space 
                       :n-calls                 200
                       :n-random-starts         50
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
    (setv (get results run-name) res)))

(plot-convergence #* results) (plt.show)
(plot-convergence #* results) (plt.savefig "compare-all-cong.svg")

;; Parallel
(defn objective [amp x y] 
  (let [msk (-> target-predicates (t-getter) (.values) (list))
        op (partial op2 amp)
        prf (-> x (op) (t-getter) (.values) (list) (np.array))
        tgt (-> y      (t-getter) (.values) (list) (np.array))
        loss (/ (np.abs (- prf tgt)) tgt)
        mask (lfor (, c p t) (zip msk prf tgt)
                   ((eval c) t p))
        cost (+ (* (np.tanh (np.abs loss)) mask) 
                  (* (- (** loss 2.0)) (np.invert mask))) ]
      (-> cost (np.nan-to-num) (np.sum) (-))))


(defn optim-run [optimizer estimator generator acquisitor]
  (let [time-stamp (-> dt (.now) (.strftime "%H%M%S-%y%m%d"))
          run-name (.format "{}-{}-{}-{}" optimizer.__name__ 
                            estimator generator acquisitor)
          res-prefix (.format "./results/{}-{}" run-name time-stamp)
          res-path (.format "{}.pkl" res-prefix)
          res-conv-path (.format "{}-conv.svg" res-prefix)
          res-objk-path (.format "{}-objk.svg" res-prefix)
          res-eval-path (.format "{}-eval.svg" res-prefix)
          amp (ac.single-ended-opamp ckt-path :pdk-path [pdk-path])
          design-objective (with-decorator (use-named-args :dimensions design-space)
                              (fn [gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1 
                                   fug-cm1 fug-cm2 fug-cm3 fug-dp1 i1 i2]
                                (let [x (np.array [gmid-cm1 gmid-cm2 gmid-cm3 gmid-dp1 
                                                   fug-cm1 fug-cm2 fug-cm3 fug-dp1 i1 i2])
                                                   y target-values]
                                  (objective amp x y))))
          t0 (time.process-time)
          res (optimizer :func                    design-objective
                         :dimensions              design-space 
                         :n-calls                 5
                         :n-random-starts         1
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
      (, run-name res)))

(for [s sweep] (optim-run #* s))













