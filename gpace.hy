(import os)
(import [datetime [datetime :as dt]])
(import [torch :as pt])
(import [gpytorch :as gpt])
(import [pandas :as pd])
(import [numpy :as np])
(import [h5py :as h5])
(import [tqdm [tqdm]])
(import [matplotlib [pyplot :as plt]])
(import [sklearn.preprocessing [MinMaxScaler minmax-scale]])
(require [hy.contrib.walk [let]])
(require [hy.contrib.loop [loop]])
(require [hy.extra.anaphoric [*]])
(import [hy.contrib.pprint [pp pprint]])

(setv data-dir "../data/ace"
      model-dir "../model/"
      op-id "op2-xh035"
      data-path f"{data-dir}/{op-id}-offset.h5"
      time-stamp (-> dt (.now) (.strftime "%d-%m-%y-%H%M%S")))

;(setv raw (pd.DataFrame 
;  (with [hdf-file (h5.File data-path "r")]
;    (dfor c (.keys hdf-file)
;      [c (->> c (get hdf-file) (np.array))]))))

(setv raw (-> data-path (pd.read-hdf) (.sample :frac 0.5)))

(setv params-x ["Wd" "Lcm2" "Ld" "Wcm2" "Lcm3" "Wcm1" "Wcm3" "Lcm1"] )
(setv params-y ["voff_stat"])

;(setv raw-shuffled (-> raw (get (+ params-x params-y)) (.sample :frac 1) (.dropna)))

(setv raw-x (-> raw (get params-x) (. values))
      raw-y (-> raw (get params-y) (. values) (np.abs) (np.log10)))

(.fit (setx scale-x (MinMaxScaler)) raw-x)
(.fit (setx scale-y (MinMaxScaler)) raw-y)

(setv train-x (-> raw-x (scale-x.transform) 
                        (pt.from-numpy) 
                        (.float) 
                        (.contiguous)
                        (.cuda)))
(setv train-y (-> raw-y (scale-y.transform) 
                        (.flatten)
                        (pt.from-numpy) 
                        (.float) 
                        (.contiguous)
                        (.cuda)))

(defclass GPModel [gpt.models.ExactGP]
  (defn __init__ [self train-x train-y likelihood]
    (.__init__ (super GPModel self) train-x train-y likelihood)
    (setv self.mean-module (gpt.means.ConstantMean)
            self.covar-module (gpt.kernels.ScaleKernel 
                                  (gpt.kernels.RBFKernel 
                                      :ard-num-dims (len params-y)))))
  (defn forward [self x]
    (gpt.distributions.MultivariateNormal (self.mean-module x) 
                                          (self.covar-module x))))

;(defclass GPModel [gpt.models.ExactGP]
;  (defn __init__ [self train-x train-y likelihood]
;    (.__init__ (super GPModel self) train-x train-y likelihood)
;    (setv self.mean-module (gpt.means.MultitaskMean 
;                              (gpt.means.ConstantMean) 
;                              :num-tasks 2))
;    (setv self.covar-module (gpt.kernels.MultitaskKernel 
;                                (gpt.kernels.RBFKernel :ard-num-dims 2) 
;                                ;(gpt.kernels.RQKernel 
;                                ;    :ard-num-dims 2 
;                                ;    :alpha-constraint (.Positive gpt.constraints)) 
;                                :num-tasks 2 :rank 1)))
;    (defn forward [self x]
;      (let [mean-x (self.mean-module x)
;            covar-x (self.covar-module x)]
;        (gpt.distributions.MultitaskMultivariateNormal mean-x covar-x)))) 
;(setv likelihood (gpt.likelihoods.MultitaskGaussianLikelihood :num-tasks 2))
;(setv model (GPModel train-x train-y likelihood))

(setv likelihood (gpt.likelihoods.GaussianLikelihood))
(setv model (GPModel train-x train-y likelihood))

(for [m [model likelihood]]
  (-> m (.cuda) (.train)))

(setv optimizer (pt.optim.Adam [ {"params" (.parameters model)} ] :lr 0.1))
(setv mll (gpt.mlls.ExactMarginalLogLikelihood likelihood model))

(setv losses
      (lfor i (setx ti (-> (setx num-iters 50) (range) (tqdm)))
        (let [_ (.zero-grad optimizer)
              output (model train-x)
              loss (- (mll output train-y))
              log-i (inc i) ]
          (.backward loss) 
          (ti.set-description (.format "Loss :{:.3}" 
                                       (setx log-loss (.item loss))))
          (.step optimizer)
          (, (inc i) log-loss))))

(for [m [model likelihood]]
  (-> m (.cpu) (.eval)))

;(os.makedirs (setx model-path f"{model-dir}/{device-name}-{time-stamp}")
;             :exist-ok True)
;(pt.save {"model" (.state-dict model) 
;          "likelihood" (.state-dict likelihood)}
;         f"{model-path}/gp-model.pt")
;(setv state-dicts (pt.load f"{model-path}/gp-model.pt"))
;(setv pt-lkh (gpt.likelihoods.MultitaskGaussianLikelihood :num-tasks 2))
;(.load-state-dict pt-lkh (get state-dicts "likelihood"))
;(.load-state-dict (setx pt-mdl (GPModel train-x train-y pt-lkh))
;                  (get state-dicts "model"))
;(for [m [pt-lkh pt-mdl]]
;  (-> m (.cpu) (.eval)))

(defclass GPModelWrapper [pt.nn.Module]
  (defn __init__ [self gp]
    (.__init__ (super))
    (setv self.gp gp))
  (defn forward [self  x]
    (as-> x it (self.gp it) (, it.mean it.variance ))))

(setv test-x (-> (pt.linspace 0 1 51) (repeat 8) (list) (pt.vstack) (. T) (.contiguous)))
(setv wrapped-model (GPModelWrapper model))

(with [_ (.no-grad pt) _ (.fast-pred-var gpt.settings) _ (.trace-mode gpt.settings)]
  (.cpu (.eval model))
  (setv fake-input test-x
        pred (wrapped-model test-x)
        traced-model (pt.jit.trace wrapped-model fake-input)))

(.save traced-model f"{model-path}/gp-trace.pt")

(setv (, traced-mean traced-var) 
      (with [(.no-grad pt)]
        
))

(setv tru (.dropna(.sort-values (get raw (& (= raw.L (np.random.choice (.unique raw.L)))
                                            (= raw.W (np.random.choice (.unique raw.W)))
                                            (= (.round raw.Vbs 2) 0.0)
                                            (= (.round raw.Vds 2) (.round raw.Vgs 2))))
                        :by ["gmid"])))

(setv tru-x (. (np.vstack [tru.gmid.values
                           (np.log10 tru.fug.values)]) T))
(setv valid-x (-> tru-x
                  (scale-x.transform) 
                  (pt.from-numpy) 
                  (.float)
                  (.contiguous)))

(with [_ (.no-grad pt) (.fast-pred-var gpt.settings)]
  (setv predictions (likelihood (pt-mdl valid-x))
        prd         (-> predictions
                        (. mean)
                        (.numpy)
                        (scale-y.inverse-transform)
                        (. T)))
  (setv (, lower 
           upper )  (.confidence-region predictions)
        lw (-> lower (.numpy) (scale-y.inverse-transform) (. T))
        up (-> upper (.numpy) (scale-y.inverse-transform) (. T))))

(setv apx (pd.DataFrame {(first params-y) (** 10 (first prd))
                         (second params-y) (second prd)}))
(setv lo (pd.DataFrame {(first params-y) (** 10 (first lw))
                        (second params-y) (second lw)}))
(setv hi (pd.DataFrame {(first params-y) (** 10 (first up))
                        (second params-y) (second up)}))

(setv (, f (, y1-ax y2-ax)) (plt.subplots 1 2 :figsize (, 8 3)))
(y1-ax.plot (-> tru (get "gmid") (. values)) 
            (-> tru (get "jd") (. values)))
(y1-ax.plot (-> tru (get "gmid") (. values)) 
            (-> apx (get "jd") (. values)) )
(y1-ax.fill-between (-> tru (get "gmid") (. values))
                    (-> lo (get "jd") (. values))
                    (-> hi (get "jd") (. values))
                    :alpha 0.5)
(y1-ax.set-yscale "log")
(y1-ax.set-xlabel "gm/Id [1/V]")
(y1-ax.set-ylabel "Jd [A/m]")
(y1-ax.legend ["Observation" "Mean" "Confidence"])
(y1-ax.set-title "Jd vs. gm/Id")
(y1-ax.grid "on")
(y2-ax.plot (-> tru (get "gmid") (. values)) 
            (-> tru (get "L") (. values)) )
(y2-ax.plot (-> tru (get "gmid") (. values)) 
            (-> apx (get "L") (. values)) )
(y2-ax.fill-between (-> tru (get "gmid") (. values))
                    (-> lo (get "L") (. values))
                    (-> hi (get "L") (. values))
                    :alpha 0.5)
(y2-ax.set-xlabel "gm/Id [1/V]")
(y2-ax.set-ylabel "L [m]")
(y2-ax.legend ["Observation" "Mean" "Confidence"])
(y2-ax.set-title "L vs. gm/Id")
(y2-ax.grid "on")
(plt.show)
