;; SpikingLab: Modelling Spiking Neural Networks with STDP learning

;; Author: Cristian Jimenez Romero - The Open University - 2015
;; Publication: SpikingLab: modelling agents controlled by Spiking Neural Networks in Netlogo
;; C Jimenez-Romero, J Johnson
;; Neural Computing and Applications 28 (1), 755-764


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;SpikingLab SNN related code:;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
breed [neurontypes neurontype]
breed [inputneurons inputneuron]
breed [normalneurons normalneuron]

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Demo: Artificial insect related code:;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
breed [testcreatures testcreature]
breed [visualsensors visualsensor]
breed [pheromonesensors pheromonesensor]

breed [sightlines sightline]
directed-link-breed [sight-trajectories sight-trajectory]

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
breed [itrails itrail]
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

directed-link-breed [normal-synapses normal-synapse]
directed-link-breed [input-synapses input-synapse]

patches-own [
  chemical             ;; amount of chemical on this patch
  food                 ;; amount of food on this patch (0, 1, or 2)
  nest?                ;; true on nest patches, false elsewhere
  nest-scent           ;; number that is higher closer to the nest
  food-source-number   ;; number (1, 2, or 3) to identify the food sources
  visual_object       ;; type (0 = empty, 1= wall, 2= obstacle, 3=food)
]

;;;
;;; Global variables including SpikingLab internals
;;;
globals [
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;SNN Module globals;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  neuron_state_open;; describe state machine of the neuron
  neuron_state_firing;; describe state machine of the neuron
  neuron_state_refractory;; describe state machine of the neuron
  excitatory_synapse
  inhibitory_synapse
  system_iter_unit ;;time steps of each iteration expressed in milliseconds
  plot-list
  plot-list2
  PulseHistoryBuffSize ;;Size of pulse history buffer
  input_value_empty ;;Indicate that there is no input value
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Demo globals;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  pspikefrequency ;;Number of spikes emitted by an input neuron in response to one stimulus
  error_free_counter ;;Number of iterations since the insect collided with a noxious stimulus
  required_error_free_iterations ;;Number of error-free iterations necessary to stop training
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;World globals;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  min_x_boundary ;;Define initial x coordinate of simulation area
  min_y_boundary ;;Define initial y coordinate of simulation area
  max_x_boundary ;;Define final x coordinate of simulation area
  max_y_boundary ;;Define final y coordinate of simulation area
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;GA Globals;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  fitness_value
  raw_weight_data
  raw_plasticity_data
  raw_delay_data
  ;;;;;;;;;;;;;;;;;;;;;;;;;;Ants Globals;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  simulation_area

]

neurontypes-own [

  neurontypeid
  ;;;;;;;;;;Neuron Dynamics;;;;;;;;;;;
  restingpotential
  firethreshold
  decayrate ;Decay rate constant
  relrefractorypotential ;Refractory potential
  intervrefractoryperiod ;Duration of absolute-refractory period
  minmembranepotential ;lowest boundary for membrane potential
  ;;;;;;;Learning Parameters;;;;;;;;;
  pos_hebb_weight ;;Weight to increase the efficacy of synapses
  pos_time_window ;; Positive learning window
  neg_hebb_weight ;;Weight to decrease the efficacy of synapses
  neg_time_window ;; negative learning window
  max_synaptic_weight ;;Maximum synaptic weight
  min_synaptic_weight ;;Minimum synaptic weight
  pos_syn_change_interval ;;ranges of pre–to–post synaptic interspike intervals over which synaptic change occur.
  neg_syn_change_interval

]


normalneurons-own [

  nlayernum
  nneuronid
  nneuronstate
  nrestingpotential
  nfirethreshold
  nmembranepotential
  ndecayrate
  nrelrefractorypotential
  nintervrefractoryperiod
  nrefractorycounter
  naxondelay
  nsynapsesarray
  nnumofsynapses
  nlast-firing-time
  nincomingspikes
  nlastspikeinput
  nneuronlabel
  nminmembranepotential
  ;;;;;;;;;;;;;;;;Learning Parameters;;;;;;;;;;;;;;;;;
  npos_hebb_weight ;;Weight to increase the efficacy of synapses
  npos_time_window ;; Positive learning window
  nneg_hebb_weight ;;Weight to decrease the efficacy of synapses
  nneg_time_window ;; negative learning window
  nmax_synaptic_weight ;;Maximum synaptic weight
  nmin_synaptic_weight ;;Minimum synaptic weight
  npos_syn_change_interval ;;ranges of pre–to–post synaptic interspike intervals over which synaptic change occur.
  nneg_syn_change_interval

]


normal-synapses-own [

  presynneuronlabel
  possynneuronlabel
  presynneuronid
  possynneuronid
  synapseefficacy
  exc_or_inh
  synapsedelay
  joblist
  learningon?

]


inputneurons-own [

   layernum
   neuronid
   neuronstate
   pulsecounter ;;Count the number of sent pulses
   interspikecounter ;;Count the time between pulses
   numberofspikes ;;Number of spikes to send
   postsynneuron
   encodedvalue
   isynapseefficacy ;;In most cases should be large enough to activate possynn with a single spike
   neuronlabel

]


testcreatures-own [

  creature_label
  creature_id
  reward_neuron
  pain_neuron
  move_neuron
  rotate_left_neuron
  rotate_right_neuron
  pheromone_neuron
  creature_sightline
  last_nest_collision
  last_food_collision
  fitness

]

pheromonesensors-own [
  sensor_id
  relative_rotation
  left_sensor_attached_to_neuron
  middle_sensor_attached_to_neuron
  right_sensor_attached_to_neuron
  attached_to_creature
]

visualsensors-own [

  sensor_id
  perceived_stimuli
  distance_to_stimuli
  relative_rotation ;;Position relative to front
  attached_to_colour
  attached_to_neuron
  attached_to_creature

]

itrails-own
[
  ttl
]

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Patches procedures:
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to setup-patches
  ask patches
  [ setup-nest
    setup-food
    recolor-patch ]
end

to setup-nest  ;; patch procedure
  if visual_object != 0 [ set nest? false stop ]
  ;; set nest? variable to true inside the nest, false elsewhere
  set nest? (distancexy 82 30) < 5
  ;; spread a nest-scent over the whole world -- stronger near the nest
  set nest-scent 200 - distancexy 82 30
end

to setup-food  ;; patch procedure
  if visual_object = 1 [ stop ]
  ;; setup food source one on the right
  if (distancexy (0.7 * max-pxcor) 53) < 5
  [ set food-source-number 1 set visual_object 3 ]
  ;; setup food source two on the lower-left
  if (distancexy (0.7 * max-pxcor) 8) < 5
  [ set food-source-number 2 set visual_object 3 ]
  ;; setup food source three on the upper-left
  if (distancexy (0.95 * max-pxcor) 30) < 5
  [ set food-source-number 3 set visual_object 3 ]
  ;; set "food" at sources to either 1 or 2, randomly
  if food-source-number > 0
  [ set food one-of [5 10] ]

end

to recolor-patch  ;; patch procedure
  if visual_object = 1 [ stop ]
  ;; give color to nest and food sources
  ifelse nest?
  [ set pcolor violet ]
  [ ifelse food > 0
    [
      set pcolor green
      ;if food-source-number = 1 [ set pcolor green ] ;cyan
      ;if food-source-number = 2 [ set pcolor green  ] ;sky
      ;if food-source-number = 3 [ set pcolor green ]
    ] ;blue
    ;; scale color to show chemical concentration
    [ set pcolor scale-color blue chemical 0.1 5 ]
  ]
end

to update-patches
  ask patches
  [ set chemical chemical * (100 - evaporation-rate) / 100  ;; slowly evaporate chemical
    recolor-patch ]
end

;;;
;;; Set global variables with their initial values
;;;
to initialize-global-vars

  set system_iter_unit 1 ;; each simulation iteration represents 1 time unit
  set neuron_state_open 1 ;;State machine idle
  set neuron_state_firing 2 ;;State machine firing
  set neuron_state_refractory 3 ;;State machine refractory
  set excitatory_synapse 1
  set inhibitory_synapse 2
  set plot-list[] ;;List for spike history
  set plot-list2[] ;;List for spike history
  set PulseHistoryBuffSize 30
  set input_value_empty -1

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;Simulation area globals;;;;;;;;;;;;;;;;;;;;;;;;
  set min_x_boundary 60  ;;Define initial x coordinate of simulation area
  set min_y_boundary 1   ;;Define initial y coordinate of simulation area
  set max_x_boundary 100 ;;Define final x coordinate of simulation area
  set max_y_boundary 60  ;;Define final y coordinate of simulation area

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Insect globals;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  set pspikefrequency 1
  set error_free_counter 0
  set required_error_free_iterations 35000

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Genetic Algorithm ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  set fitness_value 0

end

;;;
;;; Create a neuron type
;;; Parameters: Type-identifier, resting potential, firing threshold, decay rate, refractory potential, duration of refractory period, lowest boundary for membrane potential
to setup-neurontype [#pneurontypeid #prestpot #pthreshold #pdecayr #prefractpot #pintrefrectp #minmembranepotential]

  create-neurontypes 1
  [
     set shape "square"
     set neurontypeid #pneurontypeid
     set restingpotential #prestpot
     set firethreshold #pthreshold
     set decayrate #pdecayr
     set relrefractorypotential #prefractpot
     set intervrefractoryperiod #pintrefrectp
     set minmembranepotential #minmembranepotential
     ht
  ]

end


;;;
;;; Set learning parameters for neuron type pneurontypeid
;;;

to set-neurontype-learning-params [ #pneurontypeid #ppos_hebb_weight #ppos_time_window #pneg_hebb_weight #pneg_time_window #pmax_synaptic_weight
                                    #pmin_synaptic_weight #ppos_syn_change_interval #pneg_syn_change_interval]

  ask neurontypes with [neurontypeid = #pneurontypeid]
  [
    set pos_hebb_weight #ppos_hebb_weight
    set pos_time_window #ppos_time_window
    set neg_hebb_weight #pneg_hebb_weight
    set neg_time_window #pneg_time_window
    set max_synaptic_weight #pmax_synaptic_weight
    set min_synaptic_weight #pmin_synaptic_weight
    set pos_syn_change_interval #ppos_syn_change_interval
    set neg_syn_change_interval #pneg_syn_change_interval
  ]

end

;;;
;;; Declare an existing neuron "pneuronid" as neuron-type "pneurontypeid"
;;;
to set-neuron-to-neurontype [ #pneurontypeid #pneuronid ] ;Call by observer


    ask neurontypes with [neurontypeid = #pneurontypeid]
    [
      ask normalneuron #pneuronid
      [
        set nrestingpotential [restingpotential] of myself
        set nmembranepotential [restingpotential] of myself
        set nfirethreshold [firethreshold] of myself
        set ndecayrate [decayrate] of myself
        set nrelrefractorypotential [relrefractorypotential] of myself
        set nintervrefractoryperiod [intervrefractoryperiod] of myself
        set nminmembranepotential [minmembranepotential] of myself

        ;;;;;;;;;;;;;;;;Learning Parameters;;;;;;;;;;;;;;;;;
        set npos_hebb_weight [pos_hebb_weight] of myself
        set npos_time_window [pos_time_window] of myself
        set nneg_hebb_weight [neg_hebb_weight] of myself
        set nneg_time_window [neg_time_window] of myself
        set nmax_synaptic_weight [max_synaptic_weight] of myself
        set nmin_synaptic_weight [min_synaptic_weight] of myself
        set npos_syn_change_interval [pos_syn_change_interval] of myself
        set nneg_syn_change_interval [neg_syn_change_interval] of myself
      ]
     ]

end


;;;
;;; Get intern identifier of the input-neuron with label = #pneuronlabel
;;;
to-report get-input-neuronid-from-label [#pneuronlabel]

  let returned_id nobody
  ask one-of inputneurons with [neuronlabel = #pneuronlabel][set returned_id neuronid]
  report returned_id

end

;;;
;;; Get intern identifier of the normal-neuron with label = #pneuronlabel
;;;
to-report get-neuronid-from-label [#pneuronlabel]

  let returned_id nobody
  ask one-of normalneurons with [nneuronlabel = #pneuronlabel][set returned_id nneuronid]
  report returned_id

end


;;;
;;; Create a new synapse between pre-synaptic neuron: #ppresynneuronlabel and post-synaptic neuron: #ppossynneuronlabel
;;;
to setup-synapse [#ppresynneuronlabel #ppossynneuronlabel #psynapseefficacy #pexc_or_inh #psyndelay #plearningon?]

   let presynneuid get-neuronid-from-label #ppresynneuronlabel
   let possynneuid get-neuronid-from-label #ppossynneuronlabel

   let postsynneu normalneuron possynneuid
   ask normalneuron presynneuid [
        create-normal-synapse-to postsynneu [
           set presynneuronlabel #ppresynneuronlabel
           set possynneuronlabel #ppossynneuronlabel
           set presynneuronid presynneuid
           set possynneuronid possynneuid
           set synapseefficacy #psynapseefficacy
           set exc_or_inh #pexc_or_inh
           set synapsedelay #psyndelay
           set joblist []
           set learningon? #plearningon?
           ifelse (#pexc_or_inh = inhibitory_synapse)
           [
             set color red
           ]
           [
             set color grey
           ]
        ]
     ]

end


;;;
;;; Process incoming pulse from input neuron
;;;
to receive-input-neuron-pulse [ #psnefficacy #pexcinh ];;called by Neuron

  if ( nneuronstate != neuron_state_refractory )
  [
      ;;Adjust membrane potential:
      ifelse ( #pexcinh = excitatory_synapse )
      [
          set nmembranepotential nmembranepotential + #psnefficacy  ;;increase membrane potential
      ]
      [
          set nmembranepotential nmembranepotential - #psnefficacy ;;decrease membrane potential
          if (nmembranepotential < nminmembranepotential) ;; Floor for the membrane potential in case of extreme inhibition
          [
             set nmembranepotential nminmembranepotential
          ]
      ]
  ]
  set nlastspikeinput ticks

end


;;;
;;; Neuron pstneuronid# processes incoming pulse from neuron #prneuronid
;;;
to receive-pulse [ #prneuronid #snefficacy #excinh #plearningon? ];;called by neuron

  if ( nneuronstate != neuron_state_refractory )
  [
      ;;Perturb membrane potential:
      ifelse ( #excinh = excitatory_synapse )
      [
          set nmembranepotential nmembranepotential + (#snefficacy * epsp-kernel) ;;increase membrane potential
      ]
      [
          set nmembranepotential nmembranepotential - (#snefficacy * ipsp-kernel) ;;decrease membrane potential
          if (nmembranepotential < nminmembranepotential) ;; Floor for the membrane potential in case of extreme inhibition
          [
             set nmembranepotential nminmembranepotential
          ]
      ]
  ]
  ;;Remember last input spike:
  set nlastspikeinput ticks
  ;; If plasticity is activated then store pulse info for further processing and apply STDP
  if (#plearningon? and istrainingmode?)
  [
     ;;Don't let incoming-pulses history-buffer grow beyond limits (delete oldest spike):
     check-pulse-history-buffer
     ;;Create list of parameters and populate it:
     let pulseinflist[]
     set pulseinflist lput #prneuronid pulseinflist ;;Add Presynaptic neuron Id
     set pulseinflist lput #snefficacy pulseinflist ;;Add Synaptic efficacy
     set pulseinflist lput #excinh pulseinflist ;;Add excitatory or inhibitory info.
     set pulseinflist lput ticks pulseinflist ;;Add arriving time
     set pulseinflist lput false pulseinflist ;;Indicate if pulse has been processed as an EPSP following a Postsynaptic spike ( Post -> Pre, negative hebbian)
     ;;Add list of parameters to list of incoming pulses:
     set nincomingspikes lput pulseinflist nincomingspikes
     ;;Apply STDP learning rule:
     apply-stdp-learning-rule
  ]

end


;;;
;;; Neuron prepares to fire
;;;
to prepare-pulse-firing ;;Called by Neurons

   ;;Remember last firing time
   set nlast-firing-time ticks
   ;;Apply learning rule and after that empty incoming-pulses history:
   apply-stdp-learning-rule
   empty-pulse-history-buffer
   ;;transmit Pulse to postsynaptic neurons:
   propagate-pulses
   ;;Set State to refractory
   set nneuronstate neuron_state_refractory
   ;;initialize counter of refractory period in number of iterations
   set nrefractorycounter nintervrefractoryperiod

end


;;;
;;; Kernel for inhibitory post-synaptic potential
;;;
to-report ipsp-kernel ;;Called by Neurons

  report 1

end


;;;
;;; Kernel for excitatory post-synaptic potential
;;;
to-report epsp-kernel ;;Called by Neurons

  report 1

end


;;;
;;; Kernel for membrane decay towards resting potential (If current membrane pot. > Resting pot.)
;;;
to-report negative-decay-kernel ;;Called by Neurons

  report  abs(nrestingpotential - nmembranepotential) * ndecayrate

end


;;;
;;; Kernel for membrane decay towards resting potential (If current membrane pot. < Resting pot.)
;;;
to-report positive-decay-kernel ;;Called by Neurons

  report abs(nrestingpotential - nmembranepotential) * ndecayrate

end


;;;
;;; Bring membrane potential towards its resting state
;;;
to decay ;;Called by Neurons
    ;;Move membrane potential towards resting potential:
    ifelse (nmembranepotential > nrestingpotential )
    [
        let expdecay negative-decay-kernel ;
        ifelse ((nmembranepotential - expdecay) < nrestingpotential)
        [
          set nmembranepotential nrestingpotential
        ]
        [
          set nmembranepotential nmembranepotential - expdecay
        ]
    ]
    [
      let expdecay positive-decay-kernel ;
      ifelse ((nmembranepotential + expdecay) > nrestingpotential)
      [
        set nmembranepotential nrestingpotential
      ]
      [
        set nmembranepotential nmembranepotential + expdecay
      ]
    ]

end


;;;
;;; Process neuron dynamics according to its machine state
;;;
to do-neuron-dynamics ;;Called by Neurons

  ifelse ( nneuronstate = neuron_state_open )
  [
    if (nmembranepotential != nrestingpotential )
    [
      ;;Check if membrane potential reached the firing threshold
      ifelse( nmembranepotential >= nfirethreshold )
      [
        prepare-pulse-firing
        set color red
      ]
      [
         ;;Move membrane potential towards resting potential:
         decay
      ]
    ]
  ]
  [
    ;;Not idle and not firing, then refractory:
    set color pink ;;Restore normal colour
    ;;Decrease timer of absolute refractory period:
    set nrefractorycounter nrefractorycounter - system_iter_unit
    ;;Set membrane potential with refractory potential:
    set nmembranepotential nrelrefractorypotential
    if ( nrefractorycounter <= 0) ;;End of absolute refractory period?
    [
      ;;Set neuron in open state:
      set nneuronstate neuron_state_open
    ]
  ]
  ;;Continue with Axonal dynamics independently of the neuron state:
  do-synaptic-dynamics

end


;;;
;;; Delete history of incoming spikes
;;;
to empty-pulse-history-buffer ;;Called by neurons

  set nincomingspikes[]

end


;;;
;;; Apply the Spike Timing Dependent Plasticity rule
;;;
to apply-stdp-learning-rule ;;Call by neurons

  ;Apply rule: Ap.exp(dt/Tp); if dt < 0; dt = prespt - postspt
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  let currspikeinfo[]
  let itemcount 0
  let deltaweight 0

  while [itemcount < ( length nincomingspikes ) ]
  [
     set currspikeinfo ( item itemcount nincomingspikes ) ;;Get spike info:   prneuronid[0], snefficacy[1], excinh[2], arrivaltime[3], processedBynegHebb[4]

     ifelse ( item 2 currspikeinfo ) = excitatory_synapse ;;Is the spike coming from an excitatory synapsis?
     [
       let deltaspike ( item 3 currspikeinfo ) - nlast-firing-time
       if ( deltaspike >= nneg_time_window and deltaspike <= npos_time_window) ;;Is spike within learning window?
       [
           ;;Calculate learning factor:
          ifelse ( deltaspike <= 0 ) ;;Increase weight
          [
            set deltaweight  npos_hebb_weight * exp(deltaspike / npos_syn_change_interval )
            ask normal-synapse  ( item 0 currspikeinfo ) nneuronid  [update-synapse-efficacy deltaweight [nmax_synaptic_weight] of myself [nmin_synaptic_weight] of myself]
          ]
          [
            if (( item 4 currspikeinfo ) = false) ;;if spike has not been processed then compute Hebb rule:
            [
              set deltaweight (- nneg_hebb_weight * exp(- deltaspike / nneg_syn_change_interval )) ;;Turn positive delta into a negative weight
              set currspikeinfo replace-item 4 currspikeinfo true ;Indicate that this pulse has already been processed as a EPSP after Postsyn neuron has fired (negative hebbian)
              set nincomingspikes replace-item itemcount nincomingspikes currspikeinfo
              ask normal-synapse  ( item 0 currspikeinfo ) nneuronid [update-synapse-efficacy deltaweight [nmax_synaptic_weight] of myself [nmin_synaptic_weight] of myself]
            ]
          ]
       ]
     ]
     [
       ;;Inhibitory Synapses: Plasticity in inhibitory synapses not implemented yet

     ]
     set itemcount itemcount + 1
  ]

end


;;;
;;; Don't store more pulses than the specified by PulseHistoryBuffSize
;;;
to check-pulse-history-buffer ;;Call by neurons

  if( length nincomingspikes > PulseHistoryBuffSize )
  [
    ;; Remove oldest pulse in the list
    set nincomingspikes remove-item 0 nincomingspikes
  ]

end


;;;
;;; Propagate pulse to all post-synaptic neurons
;;;
to propagate-pulses ;;Call by neurons

  ;; Insert a new pulse in all synapses having the current neuron as presynaptic
  ask my-out-normal-synapses
  [
    add-pulse-job
  ]

end


;;;
;;; Process synaptic dynamics of synapses with pre-synaptic neuron: presynneuronid
;;;
to do-synaptic-dynamics ;;Call by neurons

  ;; Process all outgoing synapses with pulses in their job-list
  ask my-out-normal-synapses with [ length joblist > 0 ]
  [
    process-pulses-queue
  ]

end


;;;
;;; Enqueue pulse in synapse
;;;
to add-pulse-job ;;Call by link (synapse)

    ;;Add a new Pulse with its delay time at the end of the outgoing-synapse joblist
    set joblist lput synapsedelay joblist

end


;;;
;;; Change synaptic weight
;;;
to update-synapse-efficacy [ #deltaweight #pmax_synaptic_weight #pmin_synaptic_weight] ;;Call by synapse

  ifelse ( synapseefficacy + #deltaweight ) > #pmax_synaptic_weight
  [
    set synapseefficacy #pmax_synaptic_weight
  ]
  [
    ifelse ( synapseefficacy + #deltaweight ) < #pmin_synaptic_weight
    [
      set synapseefficacy #pmin_synaptic_weight
    ]
    [
      set synapseefficacy synapseefficacy + #deltaweight
    ]
  ]

end

;;;
;;; For each traveling pulse in synapse check if pulse has already arrived at the post-synaptic neuron
;;;
to process-pulses-queue ;;Call by synapse

  set joblist map [ i -> i - 1  ] joblist ;;Decrease all delay counters by 1 time-unit

  foreach filter [i -> i <= 0] joblist ;;Process pulses that reached the postsynaptic neuron
  [
       ;;Transmit Pulse to Postsyn Neuron:
       ask other-end [receive-pulse [presynneuronid] of myself [synapseefficacy] of myself [exc_or_inh] of myself [learningon?] of myself]
  ]
  ;;Keep only "traveling" pulses in the list :
  ;;set joblist filter [? > 0] joblist
  set joblist filter [i -> i > 0] joblist

end


;;;
;;; Create one input neuron and attach it to neuron with label #ppostsynneuronlabel (input neurons have one connection only)
;;;
to setup-input-neuron [#pposx #pposy #label #ppostsynneuronlabel #psynapseefficacy #pcoding #pnumofspikes]

  let postsynneuronid get-neuronid-from-label #ppostsynneuronlabel
  set-default-shape inputneurons "square"
  create-inputneurons 1
  [
     set layernum 0
     set neuronid who
     set neuronstate neuron_state_open
     set pulsecounter 0
     set interspikecounter 0
     set numberofspikes #pnumofspikes
     set postsynneuron postsynneuronid
     set encodedvalue input_value_empty
     set isynapseefficacy #psynapseefficacy
     setxy #pposx #pposy
     set color green
     set label #label
     set neuronlabel #label
     setup-input-synapse
  ]

end


;;;
;;; Process pulses in input neuron
;;;
to do-input-neuron-dynamics ;;Called by inputneurons

  if ( pulsecounter > 0 ) ;;process only if input-neuron has something to do
  [
    set interspikecounter interspikecounter + 1
    if (interspikecounter > encodedvalue)
    [
    ;;Transmit pulse to Post-synaptic Neuron;
      ask out-input-synapse-neighbors [receive-input-neuron-pulse [isynapseefficacy] of myself [excitatory_synapse] of myself]
      set interspikecounter 0
      set pulsecounter pulsecounter - 1
    ]
  ]

end


;;;
;;; Encode input value (integer number) into pulses
;;;
to-report set-input-value [#pencodedvalue] ;;Called by inputneurons

  ;;Check if input neuron is ready to receive input
  let inputready false
  if ( pulsecounter = 0 )
  [
    set encodedvalue #pencodedvalue
    set pulsecounter numberofspikes ;;Initialize counter with the number of pulses to transmit with the encoded value
    set interspikecounter 0
    set inputready true
  ]
  report inputready
end


;;;
;;; Ask input neuron with id = #pneuronid to accept and encode a new input value
;;;
to feed-input-neuron [#pneuronid #pencodedvalue];;Called by observer

   ask inputneuron #pneuronid
   [
     let inputready set-input-value #pencodedvalue
   ]

end


;;;
;;; Ask input neuron with label = #pneuronlabel to accept and encode a new input value
;;;
to feed-input-neuron_by_label [#pneuronlabel #pencodedvalue];;Called by observer

   ask one-of inputneurons with [ neuronlabel = #pneuronlabel ]
   [
     let inputready set-input-value #pencodedvalue
   ]

end


;;;
;;; Create link to represent synapse from input neuron to post-synaptic neuron: postsynneuron
;;;
to setup-input-synapse ;;Call from inputneurons

   let psnneuron postsynneuron
   let postsynneu one-of (normalneurons with [nneuronid = psnneuron])
   create-input-synapse-to postsynneu

end


;;;
;;; Create and initialize neuron
;;;
to setup-normal-neuron [#playernum #pposx #pposy #label #pneurontypeid]

  set-default-shape normalneurons "circle"
  let returned_id nobody
  create-normalneurons 1
  [
     set nlayernum #playernum
     set nneuronid who
     set nneuronstate neuron_state_open
     set nrefractorycounter 0
     set nincomingspikes[]
     set nnumofsynapses 0
     set nlastspikeinput 0
     setxy #pposx #pposy
     set color pink
     set label #label
     set nneuronlabel #label
     set returned_id nneuronid
  ]
  set-neuron-to-neurontype #pneurontypeid returned_id

end


;;;
;;; Process neural dynamics
;;;
to do-network-dynamics

 ask inputneurons [ do-input-neuron-dynamics ] ;Process input neurons in random order
 ask normalneurons [do-neuron-dynamics] ;Process normal neurons in random order

end


;;;
;;; Show information about neuron with id: #srcneuron
;;;
to show-neuron-info [#srcneuron] ;;Called by observer

   ;set plot-list lput ( [nmembranepotential] of one-of normalneurons with [nneuronid = neuronid_monitor1] ) plot-list
   ask normalneurons with [nneuronid = #srcneuron]
   [
      print ""
      write "Neuron Id:" write nneuronid print ""
      write "Layer " write nlayernum print ""
      write "Membrane potential: " write nmembranepotential print ""
      write "Spike threshold: " write nfirethreshold print ""
      write "Resting potential: " write nrestingpotential print ""
      write "Refractory potential: " write nrelrefractorypotential print ""
      write "Last input-pulse received at:" write nlastspikeinput print ""
      write "Last spike fired at: " write nlast-firing-time print ""
      write "Current state: " write (ifelse-value ( nneuronstate = neuron_state_open ) ["idle"] ["refractory"] ) print ""
      print ""
   ]

end


;;;
;;; Show information about synapse with pre-synaptic neuron: #srcneuron and post-synaptic neuron: #dstneuron
;;;
to show-synapse-info-from-to [#srcneuron #dstneuron] ;;Called by observer

   ask normal-synapses with [presynneuronid = #srcneuron and possynneuronid = #dstneuron]
   [
      print "Synapse from:"
      write "Neuron " write #srcneuron write " to neuron " write #dstneuron print ""
      write "Weight: " write synapseefficacy print ""
      write "Delay: " write synapsedelay write "iteration(s)" print ""
      write "Excitatory or Inhibitory: "
      write (ifelse-value ( exc_or_inh = excitatory_synapse ) ["Excitatory"] ["Inhibitory"] )
      print ""
   ]

end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;End of SNN code;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;
;;; Create insect agent
;;;
to-report create-creature [#xpos #ypos #creature_label #reward_neuron_label #pain_neuron_label #move_neuron_label #rotate_left_neuron_label #rotate_right_neuron_label #pheromone_neuron]

  let reward_neuron_id get-input-neuronid-from-label #reward_neuron_label
  let pain_neuron_id get-input-neuronid-from-label #pain_neuron_label
  let move_neuron_id get-neuronid-from-label #move_neuron_label
  let rotate_left_neuron_id get-neuronid-from-label #rotate_left_neuron_label
  let rotate_right_neuron_id get-neuronid-from-label #rotate_right_neuron_label
  let pheromone_neuron_id get-neuronid-from-label #pheromone_neuron
  let returned_id nobody
  create-testcreatures 1 [
    set shape "bug"
    setxy #xpos #ypos
    set size 3
    set color yellow
    set creature_label #creature_label
    set reward_neuron reward_neuron_id
    set pain_neuron pain_neuron_id
    set move_neuron move_neuron_id
    set rotate_left_neuron rotate_left_neuron_id
    set rotate_right_neuron rotate_right_neuron_id
    set pheromone_neuron pheromone_neuron_id
    set creature_id who
    set returned_id creature_id
    set last_nest_collision 0
    set last_food_collision 0
    set fitness 0
  ]
  report  returned_id

end

to check-creature-on-nest-or-food
  let food_amount 0
  let on_nest? false
  ask patch-here [ set food_amount food set on_nest? nest?]
  ifelse on_nest? [
    set last_nest_collision ticks
  ]
  [
    if food_amount > 0 [
      set last_food_collision ticks
      set food food - 1
      if food = 0 [ set visual_object 0 ]
    ]
  ]
end

;;;
;;; Check if creature returned food to nest and update its fitness
;;;
to compute-creature-fitness
  if last_nest_collision > last_food_collision and last_food_collision > 0 [
    set fitness fitness + ( 1000 / (last_nest_collision - last_food_collision) )
    set last_nest_collision 0
    set last_food_collision 0
  ]
end

to compute-swarm-fitness
  ;set fitness_value 0
  ask testcreatures [ set fitness_value fitness_value + fitness]
end

;;;
;;; Create photoreceptor and attach it to insect
;;;
to create-visual-sensor [ #psensor_id #pposition #colour_sensitive #attached_neuron_label #attached_creature] ;;Called by observer

  let attached_neuron_id get-input-neuronid-from-label #attached_neuron_label
  create-visualsensors 1 [
     set sensor_id #psensor_id
     set relative_rotation #pposition ;;Degrees relative to current heading - Left + Right 0 Center
     set attached_to_colour #colour_sensitive
     set attached_to_neuron attached_neuron_id
     set attached_to_creature #attached_creature
     ht
  ]

end

;;;
;;; Create pheromonereceptor and attach it to insect
;;;
to create-pheromone-sensor [ #psensor_id #pposition #left_sensor_attached_neuron_label #middle_sensor_attached_neuron_label #right_sensor_attached_neuron_label #attached_creature] ;;Called by observer

  let left_sensor_attached_neuron_id get-input-neuronid-from-label #left_sensor_attached_neuron_label
  let middle_sensor_attached_neuron_id get-input-neuronid-from-label #middle_sensor_attached_neuron_label
  let right_sensor_attached_neuron_id get-input-neuronid-from-label #right_sensor_attached_neuron_label
  create-pheromonesensors 1 [
     set sensor_id #psensor_id
     set relative_rotation #pposition ;;Degrees relative to current heading - Left + Right 0 Center
     set left_sensor_attached_to_neuron left_sensor_attached_neuron_id
     set middle_sensor_attached_to_neuron middle_sensor_attached_neuron_id
     set right_sensor_attached_to_neuron right_sensor_attached_neuron_id
     set attached_to_creature #attached_creature
     ht
  ]

end

;;;
;;; Ask pheromonereceptor if there is pheromone nearby
;;;
to sense-pheromone
  ;;;;;;;;;;;;;;;Take same position and heading of creature:;;;;;;;;;;;;;;;
  let creature_px 0
  let creature_py 0
  let creature_heading 0
  ask  testcreature attached_to_creature [set creature_px xcor set creature_py ycor set creature_heading heading];
  set xcor creature_px
  set ycor creature_py
  set heading creature_heading
  rt relative_rotation
  fd 1
  ;;;;;;;;;;;;;;;Sense
  let scent-ahead chemical-scent-at-angle   0
  let scent-right chemical-scent-at-angle  45
  let scent-left  chemical-scent-at-angle -45
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ifelse (scent-right > scent-ahead) and (scent-right > scent-left)
  [
    ;;activate right neuron
    feed-input-neuron right_sensor_attached_to_neuron 1
  ]
  [
    ifelse (scent-left > scent-ahead) and (scent-left > scent-right)
    [
      ;;activate left neuron
      feed-input-neuron left_sensor_attached_to_neuron 1
    ]
    [
      ;;activate middle neuron
      feed-input-neuron middle_sensor_attached_to_neuron 1
    ]
  ]
end

;;;
;;; Check amount of pheromone at [angle], called by pheromonereceptor
;;;
to-report chemical-scent-at-angle [angle]
  let p patch-right-and-ahead angle 1
  if p = nobody [ report 0 ]
  report [chemical] of p
end

;;;
;;; Ask photoreceptor if there is a patch ahead (within insect_view_distance) with a perceivable colour (= attached_to_colour)
;;;
to view-world-ahead_old ;;Called by visualsensors

  let itemcount 0
  let foundobj black
  ;;;;;;;;;;;;;;;Take same position and heading of creature:;;;;;;;;;;;;;;;
  let creature_px 0
  let creature_py 0
  let creature_heading 0
  ask  testcreature attached_to_creature [set creature_px xcor set creature_py ycor set creature_heading heading];
  set xcor creature_px
  set ycor creature_py
  set heading creature_heading
  rt relative_rotation
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  let view_distance insect_view_distance
  let xview 0
  let yview 0
  while [itemcount <= view_distance and foundobj = black]
  [
    set itemcount itemcount + 0.34;1
    ask patch-ahead itemcount [
       set foundobj pcolor
       set xview pxcor
       set yview pycor
    ]

  ]
end

;;;
;;; Ask photoreceptor if there is a patch ahead (within insect_view_distance) with a perceivable colour (= attached_to_colour)
;;;
to view-world-ahead ;;Called by visualsensors

  let itemcount 0
  let foundobj 0
  ;;;;;;;;;;;;;;;Take same position and heading of creature:;;;;;;;;;;;;;;;
  let creature_px 0
  let creature_py 0
  let creature_heading 0
  ask  testcreature attached_to_creature [set creature_px xcor set creature_py ycor set creature_heading heading];
  set xcor creature_px
  set ycor creature_py
  set heading creature_heading
  rt relative_rotation
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  let view_distance insect_view_distance
  let xview 0
  let yview 0
  while [itemcount <= view_distance and foundobj = 0]
  [
    set itemcount itemcount + 0.34;1
    ask patch-ahead itemcount [
       set foundobj visual_object
       set xview pxcor
       set yview pycor
    ]

  ]

  update-creature-sightline-position attached_to_creature xview yview
  ifelse (foundobj = attached_to_colour) ;;Found perceivable colour?
  [
    set distance_to_stimuli itemcount
    set perceived_stimuli foundobj
  ]
  [
    set distance_to_stimuli 0
    set perceived_stimuli 0
  ]

end

;;;
;;; Process Nociceptive, reward and visual sensation
;;;
to perceive-world ;;Called by testcreatures

 let nextobject 0
 let distobject 0
 let onobject 0
 ;; Get color of current position
 ask patch-here [ set onobject visual_object ] ;;visual_object 1:  type (1= wall, 2= obstacle, 3=food)
 ifelse (onobject = 1)
 [
    ifelse (noxious_white) ;;is White attached to a noxious stimulus
    [
      feed-input-neuron pain_neuron 1 ;;induce Pain
      update-fitness -5.5
      if (istrainingmode?)
      [
        ;;During training phase move the creature forward to avoid infinite rotation
        ;move-creature 0.5 ;;
        set error_free_counter 0
      ]
    ]
    [
        update-fitness 1.0
        feed-input-neuron reward_neuron 1 ;;induce happiness
        ask patch-here [ set pcolor black ] ;;Eat patch
    ]
 ]
 [
    ifelse (onobject = 2)
    [
      ifelse (noxious_red) ;;is Red attached to a noxious stimulus
      [
        update-fitness -5.5
        feed-input-neuron pain_neuron 1 ;;induce Pain
        if (istrainingmode?)
        [
          ;;During training phase move the creature forward to avoid infinite rotation
          ;move-creature 0.5
          set error_free_counter 0
        ]
      ]
      [
          update-fitness 1.0
          feed-input-neuron reward_neuron 1 ;;induce happiness
          ask patch-here [ set pcolor black ]  ;;Eat patch
      ]
    ]
    [
      if (onobject = 3)
      [
         ifelse (noxious_green) ;;is Green attached to a noxious stimulus
         [
           update-fitness -5.5
           feed-input-neuron pain_neuron 1 ;;induce Pain
           if (istrainingmode?)
           [
             ;;During training phase move the creature forward to avoid infinite rotation
             ;move-creature 0.5
             set error_free_counter 0
           ]
         ]
         [
             update-fitness 1.0
             feed-input-neuron reward_neuron 1 ;;induce happiness
             ask patch-here [ set pcolor black ]  ;;Eat patch
         ]
      ]
    ]
 ]
 ask visualsensors [propagate-visual-stimuli]

end

;;;
;;; Process Nociceptive, reward and visual sensation
;;;
to perceive-world_old ;;Called by testcreatures

 let nextobject 0
 let distobject 0
 let onobject 0
 ;; Get color of current position
 ask patch-here [ set onobject pcolor ] ;;visual_object 1:  type (1= wall, 2= obstacle, 3=food)
 ifelse (onobject = white)
 [
    ifelse (noxious_white) ;;is White attached to a noxious stimulus
    [
      feed-input-neuron pain_neuron 1 ;;induce Pain
      update-fitness -1
      if (istrainingmode?)
      [
        ;;During training phase move the creature forward to avoid infinite rotation
        ;move-creature 0.5 ;;
        set error_free_counter 0
      ]
    ]
    [
        update-fitness 1
        feed-input-neuron reward_neuron 1 ;;induce happiness
        ask patch-here [ set pcolor black ] ;;Eat patch
    ]
 ]
 [
    ifelse (onobject = red)
    [
      ifelse (noxious_red) ;;is Red attached to a noxious stimulus
      [
        update-fitness -1
        feed-input-neuron pain_neuron 1 ;;induce Pain
        if (istrainingmode?)
        [
          ;;During training phase move the creature forward to avoid infinite rotation
          ;move-creature 0.5
          set error_free_counter 0
        ]
      ]
      [
          update-fitness 1
          feed-input-neuron reward_neuron 1 ;;induce happiness
          ask patch-here [ set pcolor black ]  ;;Eat patch
      ]
    ]
    [
      if (onobject = green)
      [
         ifelse (noxious_green) ;;is Green attached to a noxious stimulus
         [
           update-fitness -1
           feed-input-neuron pain_neuron 1 ;;induce Pain
           if (istrainingmode?)
           [
             ;;During training phase move the creature forward to avoid infinite rotation
             ;move-creature 0.5
             set error_free_counter 0
           ]
         ]
         [
             update-fitness 1
             feed-input-neuron reward_neuron 1 ;;induce happiness
             ask patch-here [ set pcolor black ]  ;;Eat patch
         ]
      ]
    ]
 ]
 ask visualsensors [propagate-visual-stimuli]

end

;;;
;;; Move or rotate according to the active motoneuron
;;;
to do-actuators ;;Called by Creature

 let dorotation_left? false
 let dorotation_right? false
 let domovement? false
 let dopheromone? false
 ;;Check rotate left actuator
 ask normalneuron rotate_left_neuron [
   if (nlast-firing-time = ticks);
   [
     set dorotation_left? true
   ]
 ]
 ;;Check rotate right actuator
 ask normalneuron rotate_right_neuron [
   if (nlast-firing-time = ticks);
   [
     set dorotation_right? true
   ]
 ]
 ;;Check move forward actuator
 ask normalneuron move_neuron[
   if (nlast-firing-time = ticks);
   [
     set domovement? true
   ]
 ]

 ;;Check drop pheromone actuator
 ask normalneuron pheromone_neuron[
   if (nlast-firing-time = ticks);
   [
     set dopheromone? true
   ]
 ]

 if (dorotation_left?)
 [
   ;set fitness_value fitness_value - 0.003
   ;print "left"
    rotate-creature -6
 ]

 if (dorotation_right?)
 [
   ;set fitness_value fitness_value - 0.003
   ;print "right"
    rotate-creature 6
 ]

 if (domovement?)
 [
   ;set fitness_value fitness_value - 0.003
   move-creature 0.6
 ]

 if (dopheromone?)
 [
   set chemical chemical + 60
 ]

end

;;;
;;; Photoreceptor excitates the connected input neuron
;;;
to propagate-visual-stimuli ;;Called by visual sensor

  if (attached_to_colour = perceived_stimuli) ;;Only produce an action potential if the corresponding associated stimulus was sensed
  [
     feed-input-neuron attached_to_neuron distance_to_stimuli;
  ]

end

;;;
;;; Move insect (#move_units) patches forward
;;;
to move-creature [#move_units]
  if (leave_trail_on?) [Leave-trail]
  fd #move_units
end

;;;
;;; Rotate insect (#rotate_units) degrees
;;;
to rotate-creature [#rotate_units]
  rt #rotate_units
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;End of insect related code;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Show a sightline indicating the patch the insect is looking at
;;;
to-report create-sightline

  let sightline_id nobody
  create-sightlines 1
  [
     set shape "circle"
     set size 0.5
     set color orange
     set sightline_id who
     ht
  ]
  report sightline_id

end

to update-creature-sightline-position [#creatureid #posx #posy]

  ifelse (show_sight_line?)
  [
    let attached_sightline 0
    ask testcreature #creatureid [set attached_sightline creature_sightline]
    ask sightline attached_sightline [setxy #posx #posy]
    ask sight-trajectories [show-link]
  ]
  [
    ask sight-trajectories [hide-link]
  ]

end


to attach-sightline-to-creature [#creature_id #sightline_id]

  let sightline_agent sightline #sightline_id
  ask sightline_agent [setxy [xcor] of testcreature #creature_id [ycor] of testcreature #creature_id]
  ask testcreature #creature_id [
    set creature_sightline #sightline_id
    create-sight-trajectory-to sightline_agent [set color orange set thickness 0.4]
  ]

end

;;;
;;; Leave a yellow arrow behind the insect indicating its heading
;;;
to leave-trail ; [posx posy]

  hatch-itrails 1
  [
     set shape "arrow"
     set size 1
     set color yellow
     set ttl 2000
  ]

end

;;;
;;; Check if it is time to remove the trail
;;;
to check-trail

  set ttl ttl - 1
  if ttl <= 0
  [
     die
  ]

end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
to-report parse_csv_line [ #csv-line ]
  let parameter_end true
  let parameters_line #csv-line
  let current_parameter ""
  let parameters_list[]
  while [ parameter_end != false ]
  [
    set parameter_end position "," parameters_line
    if ( parameter_end != false ) [
      set current_parameter substring parameters_line 0 parameter_end
      set parameters_line substring parameters_line ( parameter_end + 1 ) ( (length parameters_line));
      ;;add current_parameter to destination list
      ;parameters_list
      set parameters_list lput (read-from-string current_parameter) parameters_list
    ]
  ]
  if ( length parameters_line ) > 0 [ ;;Get last parameter
    ;;add current_parameter to destination list
    set parameters_list lput (read-from-string parameters_line) parameters_list
  ]

 report parameters_list
end


to read_l2l_config_old [ #filename ]
  file-close-all
  ;file-open "eeg_data_offline_1.txt"
  carefully [
    file-open #filename ;"D:\\Research\\SNN_IRIS_DATASET\\iris.csv"
    let weight_data ( parse_csv_line file-read-line )
    ;let plasticity_data parse_csv_line file-read-line
    let delay_data parse_csv_line file-read-line
    ;;;; build_neural_circuit weight_data plasticity_data delay_data
  ]
  [
    print error-message
  ]

end

to read_l2l_config [ #filename ]
  let weight_data[]
  let plasticity_data[]
  let delay_data[]
  file-close-all
  carefully [
    file-open #filename
    set raw_weight_data file-read-line
    ;set raw_plasticity_data file-read-line
    set raw_delay_data file-read-line
    set weight_data ( parse_csv_line raw_weight_data )
    ;let plasticity_data ( parse_csv_line raw_plasticity_data )
    set delay_data ( parse_csv_line raw_delay_data )
  ]
  [
    print error-message
  ]
  setup_neural_parameters
  ;;;;;;;;;;;;;;;;;;;;;
  ;; load test data here...
  ;;;;;;;;;;;;;;;;;;;;;
  let n_insects 10
  let insect_offset 0
  repeat n_insects [
    build_neural_circuit insect_offset weight_data plasticity_data delay_data
    set insect_offset insect_offset + 1000
  ]

end

to load_l2l_parameters_copy
  let weight_data[]
  let plasticity_data[]
  let delay_data[]
  let current_index 1
  ;;;;;;
  ;file-open "testdata.txt"
  ;;;;;;
  repeat 220 [
    set weight_data lput (runresult (word "weight" current_index)) weight_data
    ;;Deactivate plasticity in this experiment, use 0 instead of: set plasticity_data lput (runresult (word "plasticity" current_index)) plasticity_data
    set plasticity_data lput 0 plasticity_data
    set delay_data lput (runresult (word "delay" current_index)) delay_data
    set current_index current_index + 1
  ]
  setup_neural_parameters

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  load_test_brain_data
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  let n_insects 10
  let insect_offset 0
  repeat n_insects [
    build_neural_circuit insect_offset weight_data plasticity_data delay_data
    set insect_offset insect_offset + 1000
  ]

end

to load_test_brain_data

set weight1      18.5
set weight2        15
set weight3     -11.5
set weight4     -11.5
set weight5     -15.5
set weight6       -15
set weight7       -11
set weight8      -9.5
set weight9         2
set weight10    -10.5
set weight11       10
set weight12       -8
set weight13      -20
set weight14      -11
set weight15     -3.5
set weight16      2.5
set weight17     16.5
set weight18      5.5
set weight19       -9
set weight20       20
set weight21        6
set weight22    -13.5
set weight23        8
set weight24       20
set weight25       20
set weight26       10
set weight27        5
set weight28       -1
set weight29     -0.5
set weight30     15.5
set weight31      -12
set weight32       18
set weight33     -7.5
set weight34     19.5
set weight35    -14.5
set weight36    -12.5
set weight37       18
set weight38        8
set weight39       -6
set weight40      -13
set weight41      -15
set weight42      -14
set weight43     -1.5
set weight44        5
set weight45       14
set weight46        5
set weight47       -1
set weight48      -19
set weight49       17
set weight50        6
set weight51    -18.5
set weight52     18.5
set weight53      -20
set weight54       20
set weight55      -11
set weight56      -10
set weight57      9.5
set weight58      0.5
set weight59     13.5
set weight60    -18.5
set weight61      3.5
set weight62       18
set weight63    -18.5
set weight64       19
set weight65      -13
set weight66      -20
set weight67     15.5
set weight68       -9
set weight69        5
set weight70      -15
set weight71       -6
set weight72     17.5
set weight73       -8
set weight74      -10
set weight75     -1.5
set weight76     10.5
set weight77     11.5
set weight78        9
set weight79     14.5
set weight80      -10
set weight81     -7.5
set weight82        3
set weight83       11
set weight84      5.5
set weight85     16.5
set weight86       -4
set weight87    -16.5
set weight88     -4.5
set weight89      0.5
set weight90     12.5
set weight91      -14
set weight92     17.5
set weight93       -1
set weight94     -1.5
set weight95     16.5
set weight96    -13.5
set weight97        6
set weight98        1
set weight99    -14.5
set weight100     6.5
set weight101      10
set weight102    -4.5
set weight103   -19.5
set weight104    -4.5
set weight105    -5.5
set weight106      -5
set weight107      -5
set weight108      10
set weight109       4
set weight110      11
set weight111      17
set weight112      -3
set weight113      16
set weight114    -6.5
set weight115     -10
set weight116   -19.5
set weight117      -3
set weight118    -7.5
set weight119       4
set weight120      17
set weight121     -19
set weight122      13
set weight123      18
set weight124     -18
set weight125    16.5
set weight126   -11.5
set weight127   -16.5
set weight128    -2.5
set weight129      -7
set weight130      20
set weight131     -20
set weight132       8
set weight133     -10
set weight134      18
set weight135   -14.5
set weight136      15
set weight137       5
set weight138     -10
set weight139      -8
set weight140      -4
set weight141    -3.5
set weight142     3.5
set weight143      -8
set weight144       1
set weight145     7.5
set weight146     -16
set weight147    12.5
set weight148     6.5
set weight149    -0.5
set weight150     -10
set weight151       7
set weight152      -9
set weight153     3.5
set weight154    11.5
set weight155      20
set weight156      15
set weight157      -6
set weight158     6.5
set weight159       7
set weight160     -13
set weight161      17
set weight162    11.5
set weight163    -5.5
set weight164       8
set weight165      -4
set weight166      -4
set weight167      -5
set weight168    -7.5
set weight169     -11
set weight170      -8
set weight171    12.5
set weight172      -9
set weight173    18.5
set weight174      -7
set weight175     -15
set weight176     -20
set weight177   -17.5
set weight178      20
set weight179     -11
set weight180    -4.5
set weight181   -11.5
set weight182    -8.5
set weight183     4.5
set weight184    -9.5
set weight185     -17
set weight186       1
set weight187     9.5
set weight188     -11
set weight189      12
set weight190     -11
set weight191    -7.5
set weight192     -16
set weight193    -7.5
set weight194       1
set weight195      11
set weight196    -8.5
set weight197       6
set weight198     3.5
set weight199     5.5
set weight200     8.5
set weight201      12
set weight202   -14.5
set weight203   -10.5
set weight204      20
set weight205     -19
set weight206     -20
set weight207      -8
set weight208       5
set weight209      12
set weight210   -10.5
set weight211      11
set weight212      13
set weight213      12
set weight214     9.5
set weight215   -14.5
set weight216     -20
set weight217      -2
set weight218    15.5
set weight219    19.5
set weight220       2
set delay1          7
set delay2          3
set delay3          3
set delay4          7
set delay5          7
set delay6          2
set delay7          2
set delay8          7
set delay9          7
set delay10         5
set delay11         7
set delay12         7
set delay13         2
set delay14         4
set delay15         2
set delay16         4
set delay17         7
set delay18         5
set delay19         6
set delay20         3
set delay21         5
set delay22         1
set delay23         2
set delay24         7
set delay25         5
set delay26         3
set delay27         1
set delay28         1
set delay29         4
set delay30         2
set delay31         7
set delay32         2
set delay33         6
set delay34         6
set delay35         7
set delay36         6
set delay37         2
set delay38         7
set delay39         3
set delay40         1
set delay41         7
set delay42         7
set delay43         4
set delay44         5
set delay45         5
set delay46         4
set delay47         4
set delay48         4
set delay49         5
set delay50         1
set delay51         5
set delay52         1
set delay53         2
set delay54         7
set delay55         2
set delay56         1
set delay57         1
set delay58         3
set delay59         4
set delay60         3
set delay61         3
set delay62         5
set delay63         4
set delay64         6
set delay65         5
set delay66         3
set delay67         3
set delay68         3
set delay69         5
set delay70         5
set delay71         2
set delay72         7
set delay73         2
set delay74         2
set delay75         2
set delay76         4
set delay77         3
set delay78         7
set delay79         1
set delay80         3
set delay81         1
set delay82         2
set delay83         6
set delay84         7
set delay85         6
set delay86         3
set delay87         5
set delay88         4
set delay89         6
set delay90         3
set delay91         5
set delay92         1
set delay93         2
set delay94         7
set delay95         6
set delay96         4
set delay97         6
set delay98         3
set delay99         4
set delay100        4
set delay101        7
set delay102        2
set delay103        5
set delay104        5
set delay105        4
set delay106        6
set delay107        5
set delay108        6
set delay109        7
set delay110        2
set delay111        3
set delay112        2
set delay113        2
set delay114        4
set delay115        1
set delay116        2
set delay117        5
set delay118        7
set delay119        2
set delay120        7
set delay121        1
set delay122        2
set delay123        7
set delay124        5
set delay125        2
set delay126        1
set delay127        4
set delay128        3
set delay129        5
set delay130        5
set delay131        4
set delay132        3
set delay133        6
set delay134        6
set delay135        2
set delay136        7
set delay137        7
set delay138        3
set delay139        2
set delay140        2
set delay141        6
set delay142        7
set delay143        5
set delay144        1
set delay145        1
set delay146        4
set delay147        1
set delay148        6
set delay149        3
set delay150        4
set delay151        6
set delay152        4
set delay153        5
set delay154        1
set delay155        7
set delay156        1
set delay157        3
set delay158        5
set delay159        2
set delay160        5
set delay161        5
set delay162        7
set delay163        1
set delay164        5
set delay165        5
set delay166        1
set delay167        6
set delay168        6
set delay169        2
set delay170        1
set delay171        3
set delay172        1
set delay173        1
set delay174        5
set delay175        5
set delay176        1
set delay177        1
set delay178        1
set delay179        1
set delay180        7
set delay181        3
set delay182        1
set delay183        4
set delay184        1
set delay185        2
set delay186        7
set delay187        5
set delay188        2
set delay189        6
set delay190        3
set delay191        4
set delay192        3
set delay193        3
set delay194        7
set delay195        7
set delay196        5
set delay197        5
set delay198        3
set delay199        2
set delay200        7
set delay201        7
set delay202        7
set delay203        5
set delay204        4
set delay205        1
set delay206        1
set delay207        3
set delay208        3
set delay209        5
set delay210        2
set delay211        2
set delay212        6
set delay213        4
set delay214        1
set delay215        2
set delay216        5
set delay217        3
set delay218        1
set delay219        3
set delay220        5

end


to load_l2l_parameters
  let weight_data[]
  let plasticity_data[]
  let delay_data[]
  let current_index 1
  ;;;;;;
  ;file-open "testdata.txt"
  ;;;;;;
  repeat 220 [
    set weight_data lput (runresult (word "weight" current_index)) weight_data
    ;;Deactivate plasticity in this experiment, use 0 instead of: set plasticity_data lput (runresult (word "plasticity" current_index)) plasticity_data
    set plasticity_data lput 0 plasticity_data
    set delay_data lput (runresult (word "delay" current_index)) delay_data
    set current_index current_index + 1
  ]
  setup_neural_parameters

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  load_test_brain_data
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  let n_insects 10
  let insect_offset 0
  repeat n_insects [
    build_neural_circuit insect_offset weight_data plasticity_data delay_data
    set insect_offset insect_offset + 1000
  ]

end

to setup_neural_parameters

  ;;;;;;;;;;;;;;Setup Neuron types;;;;;;;;;;;;;;
  ;;;;;;;;;;;;;;;;Neuron type1:
  setup-neurontype 1 -65 -55 0.08 -75 3 -75
  set-neurontype-learning-params 1 0.045 75 0.045 -50 15 1 (8 + 3) (15 + 3) ;[ #pneurontypeid #ppos_hebb_weight #ppos_time_window #pneg_hebb_weight #pneg_time_window #pmax_synaptic_weight
                                    ;#pmin_synaptic_weight #ppos_syn_change_interval #pneg_syn_change_interval]


  ;;;;;;;;;;;;;;;;Neuron type2:
  setup-neurontype 2 -65 -55 0.08 -70 1 -70 ;(typeid restpot threshold decayr refractpot refracttime)
  set-neurontype-learning-params 2 0.09 55 0.09 -25 15 1 8 15

end

to build_neural_circuit [ #insect_offset #weight_data #plasticity_data #delay_data ]


  ;;;;;;;;;;;;;;Layer 1: Visual Afferent neurons with receptors ;;;;;;;;;;;;;;;
  setup-normal-neuron 1 15 10  (#insect_offset + 11) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron 5 10  (#insect_offset + 1)  (#insect_offset + 11) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron psynapseefficacy pcoding pnumofspikes]

  setup-normal-neuron 1 15 15  (#insect_offset + 12) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron    5 15  (#insect_offset + 2)  (#insect_offset + 12) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron ipsynapseefficacy pcoding pnumofspikes]

  ;setup-normal-neuron 1 15 20  (#insect_offset + 13) 1 ;;[layernum pposx pposy pid pneurontypeid]
  ;setup-input-neuron    5 20  (#insect_offset + 3)  (#insect_offset + 13) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron ipsynapseefficacy pcoding pnumofspikes]

  ;;;;;;;;;;;;;;Layer 1b: Pheromone Afferent neurons with receptors ;;;;;;;;;;;;;;;
  ;;Left
  setup-normal-neuron 1 15 20  (#insect_offset + 13) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron    5 20  (#insect_offset + 3)  (#insect_offset + 13) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron ipsynapseefficacy pcoding pnumofspikes]
  ;;Front
  setup-normal-neuron 1 15 25  (#insect_offset + 14) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron    5 25  (#insect_offset + 4)  (#insect_offset + 14) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron ipsynapseefficacy pcoding pnumofspikes]
  ;;Right
  setup-normal-neuron 1 15 30  (#insect_offset + 15) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron    5 30  (#insect_offset + 5)  (#insect_offset + 15) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron ipsynapseefficacy pcoding pnumofspikes]
  ;;;;;;;;;;;;;;; Layer 1c: nociceptive and reward neurons with receptors ;;;;;;;;;;;;;;;;;;;;;;

  ;Nociceptive pathway:
  setup-normal-neuron 1 15 35  (#insect_offset + 16) 1
  setup-input-neuron   5 35  (#insect_offset + 6)  (#insect_offset + 16) 20 1 pspikefrequency

  ;Reward pathway:
  setup-normal-neuron 1 15 40  (#insect_offset + 17) 1
  setup-input-neuron   5 40  (#insect_offset + 7)  (#insect_offset + 17) 20 1 pspikefrequency


  ;;;;;;Create heart layer 1d:
  setup-normal-neuron 2 15 50  (#insect_offset + 18) 2
  setup-normal-neuron 2 15 55  (#insect_offset + 19) 2
  setup-synapse  (#insect_offset + 18)  (#insect_offset + 19) 15 excitatory_synapse 2 false ;(no plasticity needed)
  setup-synapse  (#insect_offset + 19)  (#insect_offset + 18) 15 excitatory_synapse 3 false ;(no plasticity needed)
  setup-input-neuron 10 50  (#insect_offset + 8)  (#insect_offset + 18) 20 1 pspikefrequency ;;Voltage clamp to start pacemaker


  ;;;;;;Create neuronal layer 2 (6):
  let neuron_counter 0
  ;; x = r cos(t) y = r sin(t)
  repeat 10 [
    setup-normal-neuron 3 (10 * cos(neuron_counter * 36) + 40) ( 10 * sin(neuron_counter * 36) + 30)  (#insect_offset + 200 + neuron_counter) 1 ;;[layernum pposx pposy pid pneurontypeid]
    set neuron_counter neuron_counter + 1
  ]

  ;;;;;;Create neuronal motor layer 3 (2):
  set neuron_counter 0
  repeat 4 [
    setup-normal-neuron 4 55 ( 22 + neuron_counter * 5)  (#insect_offset + 300 + neuron_counter) 1 ;;[layernum pposx pposy pid pneurontypeid]
    set neuron_counter neuron_counter + 1
  ]


  ;;;;;;;;;;;;;;;;;;;;;;Create Creature and attach neural circuit
  ;; Start insect hearth
  feed-input-neuron_by_label  (#insect_offset + 8) 1

  let creatureid create-creature 82 30 (#insect_offset + 1)  (#insect_offset + 4)  (#insect_offset + 3)  (#insect_offset + 300)  (#insect_offset + 301) (#insect_offset + 302) (#insect_offset + 303);;[#xpos #ypos #creature_id #reward_input_neuron #pain_input_neuron #move_neuron #rotate_neuron_left #rotate_neuron_right ]
  ;;;;;;;;;;Create Pheromone sensor
  ;create-pheromone-sensor [ #psensor_id #pposition #left_sensor_attached_neuron_label #middle_sensor_attached_neuron_label #right_sensor_attached_neuron_label #attached_creature]
  create-pheromone-sensor (#insect_offset + 1) 0 (#insect_offset + 5) (#insect_offset + 6) (#insect_offset + 7) creatureid

  ;;;;;;;;;;Create Visual sensors and attach neurons to them;;;;;;;;;
  create-visual-sensor  (#insect_offset + 1) 0  1  (#insect_offset + 1) creatureid;[ psensor_id pposition colour_sensitive attached_neuron attached_creature]
  create-visual-sensor  (#insect_offset + 2) 0 2 (#insect_offset + 2) creatureid;[ psensor_id pposition colour_sensitive attached_neuron attached_creature]
  ;create-visual-sensor  (#insect_offset + 3) 0 3 (#insect_offset + 3) creatureid;[ psensor_id pposition colour_sensitive attached_neuron attached_creature]

  ;;;;;;;;;;Create Sightline ;;;;;;;;;;;;;
  let sightlineid create-sightline
  attach-sightline-to-creature creatureid sightlineid

  ;; Activate training mode
  set istrainingmode? true

  ;;;;;;;;;;;;;;;;;;;;Create synapses from layer 1 to layer 2:
  let type_exc_or_inh_synapse excitatory_synapse
  let current_index 0

  let layer_from 0
  let layer_to 0
  repeat 8 [
    set layer_to 0
    repeat 10 [
      let current_weight item current_index #weight_data
      let current_delay item current_index #delay_data
      ;let current_plasticity item current_index #plasticity_data
      let current_plasticity 0
      ifelse current_weight >= 0 [
        set type_exc_or_inh_synapse excitatory_synapse
      ]
      [
        set type_exc_or_inh_synapse inhibitory_synapse
      ]

      let plasticity? false
      if current_plasticity = 1
      [
        set plasticity? true
      ]

      setup-synapse  (#insect_offset + 11 + layer_from)  (#insect_offset + 200 + layer_to) abs(current_weight) type_exc_or_inh_synapse current_delay plasticity?
      set current_index current_index + 1
      set layer_to layer_to + 1
    ]
    set layer_from layer_from + 1
  ]

  ;;;;;;;;;;;;;;;;;;;;Create synapses from layer 2 to layer 2:
  set layer_from 0
  set layer_to 0
  repeat 10 [
    set layer_to 0
    repeat 10 [
      let current_weight item current_index #weight_data
      let current_delay item current_index #delay_data
      ;let current_plasticity item current_index #plasticity_data
      let current_plasticity 0
      ifelse current_weight >= 0 [
        set type_exc_or_inh_synapse excitatory_synapse
      ]
      [
        set type_exc_or_inh_synapse inhibitory_synapse
      ]

      let plasticity? false
      if current_plasticity = 1
      [
        set plasticity? true
      ]
      if layer_from != layer_to [
        setup-synapse  (#insect_offset + 200 + layer_from)  (#insect_offset + 200 + layer_to) abs(current_weight) type_exc_or_inh_synapse current_delay plasticity?
        set current_index current_index + 1
      ]
      set layer_to layer_to + 1
    ]
    set layer_from layer_from + 1
  ]

  ;;;;;;;;;;;;;;;;;;;;Create synapses from layer 2 to layer 3:
  set layer_from 0
  set layer_to 0
  repeat 10 [
    set layer_to 0
    repeat 4 [
      let current_weight item current_index #weight_data
      let current_delay item current_index #delay_data
      ;let current_plasticity item current_index #plasticity_data
      let current_plasticity 0
      ifelse current_weight >= 0 [
        set type_exc_or_inh_synapse excitatory_synapse
      ]
      [
        set type_exc_or_inh_synapse inhibitory_synapse
      ]

      let plasticity? false
      if current_plasticity = 1
      [
        set plasticity? true
      ]

      setup-synapse  (#insect_offset + 200 + layer_from)  (#insect_offset + 300 + layer_to) abs(current_weight) type_exc_or_inh_synapse current_delay plasticity?
      set current_index current_index + 1
      set layer_to layer_to + 1
    ]
    set layer_from layer_from + 1
  ]

end

to update-fitness [ #fitness_change ]
  set fitness_value fitness_value + #fitness_change
end

;;;
;;; Create neural circuit, insect and world
;;;
to setup

  clear-all

  RESET-TICKS
  random-seed 7850
  initialize-global-vars
  ;;;;;;;;;;Draw world with white, green and red patches;;;;;;;;
  draw-world
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  let config_file "individual_config.csv"
  read_l2l_config config_file ;"individual01.csv"
  ;load_l2l_parameters

end

;;;
;;; Generate insect world with 3 types of patches
;;;
to draw-world
   ask patches [set visual_object 0 set nest? false set food 0]
   ;;;;;;;;;;Create a grid of white patches representing walls in the virtual worls
   ask patches with [ pxcor >= min_x_boundary and pycor = min_y_boundary and pxcor <= max_x_boundary ] [ set pcolor red  ]
   ask patches with [ pxcor >= min_x_boundary and pycor = max_y_boundary ] [ set pcolor red ]
   ask patches with [ pycor >= min_y_boundary and pxcor = min_x_boundary and pxcor <= max_x_boundary] [ set pcolor red ]
   ask patches with [ pycor >= min_y_boundary and pxcor = max_x_boundary ] [ set pcolor red ]
   let ccolumns 0
   while [ ccolumns < 20 ]
   [
     set ccolumns ccolumns + 3
     ;ask patches with [ pycor >= (min_y_boundary + 3) and pycor <= (max_y_boundary + 15) and pxcor = (min_x_boundary + 2) + ccolumns * 2 ] [ set pcolor red ]
     let column_start 5
     repeat 5 [
       ;;ask patches with [ pycor >= (column_start) and pycor <= (column_start + 5) and pxcor = (min_x_boundary + 2) + ccolumns * 2 ] [ set pcolor red ]
       set column_start column_start + 12
     ]

   ]
   ask patches with [ pxcor > min_x_boundary and pxcor < max_x_boundary and pycor > 1 and pycor < max_y_boundary ]
   [
     let worldcolor random(10)

     if (worldcolor >= 1 and worldcolor < 5)
     [
       set pcolor black
       set visual_object 0
       ;set pcolor red
     ]
   ]

  set simulation_area patches with [ pxcor >= min_x_boundary and pxcor <= max_x_boundary and pycor >= min_y_boundary and pycor <= max_y_boundary ]
  ask simulation_area [set visual_object 0 set nest? false]
  ask simulation_area with [ pcolor = red ][set visual_object 1]       ;; type (1= wall, 2= obstacle, 3=food)
  ;ask simulation_area with [ pcolor = red ][set visual_object 2]
  ;ask simulation_area with [ pcolor = green ][set visual_object 3]
  ask simulation_area [ setup-food setup-nest ]

end

;;;
;;; Generate insect world with 3 types of patches
;;;
to draw-world_old

   ;;;;;;;;;;Create a grid of white patches representing walls in the virtual worls
   ask patches with [ pxcor >= min_x_boundary and pycor = min_y_boundary and pxcor <= max_x_boundary ] [ set pcolor white  ]
   ask patches with [ pxcor >= min_x_boundary and pycor = max_y_boundary ] [ set pcolor white ]
   ask patches with [ pycor >= min_y_boundary and pxcor = min_x_boundary and pxcor <= max_x_boundary] [ set pcolor white ]
   ask patches with [ pycor >= min_y_boundary and pxcor = max_x_boundary ] [ set pcolor white ]
   let ccolumns 0
   while [ ccolumns < 20 ]
   [
     set ccolumns ccolumns + 3
     ask patches with [ pycor >= (min_y_boundary + 3) and pycor <= (max_y_boundary + 15) and pxcor = (min_x_boundary + 2) + ccolumns * 2 ] [ set pcolor white ]
   ]
   ask patches with [ pxcor > min_x_boundary and pxcor < max_x_boundary and pycor > 1 and pycor < max_y_boundary ]
   [
     let worldcolor random(10)

     ifelse (worldcolor = 1)
     [
       set pcolor red
     ]
     [
       if (worldcolor >= 2 and worldcolor < 3)
       [
         set pcolor green
       ]
     ]
   ]

  ask patches [set visual_object 0 ]
  ask patches with [ pcolor = white ][set visual_object 1]       ;; type (1= wall, 2= obstacle, 3=food)
  ask patches with [ pcolor = red ][set visual_object 2]
  ask patches with [ pcolor = green ][set visual_object 3]

end

;;;
;;; Don't allow the insect to go beyond the world boundaries
;;;
to check-boundaries

   if (istrainingmode?)
   [
      set error_free_counter error_free_counter + 1
      if(error_free_counter > required_error_free_iterations)
      [
       set istrainingmode? false
      ]
      ask testcreatures [
        if (xcor < min_x_boundary or xcor > max_x_boundary) or (ycor < min_y_boundary or ycor > max_y_boundary)
        [
          setxy 82 30
        ]
      ]
   ]

end

to write_l2l_results [ #filename ]
  file-close-all
  file-open #filename
  file-print raw_weight_data
  file-print raw_plasticity_data
  file-print raw_delay_data
  file-print fitness_value
  file-flush
  file-close-all

end

;;;
;;; Run simulation
;;;
to go

 if (awakecreature?)
 [
   ask itrails [ check-trail ]
   ask visualsensors [ view-world-ahead ] ;;Feed visual sensors at first
   ask pheromonesensors [ sense-pheromone ] ;;Sniff nearby pheromone
   ask testcreatures [ perceive-world]; integrate visual and touch information
   do-network-dynamics
   ask testcreatures [do-actuators check-creature-on-nest-or-food compute-creature-fitness]
 ]
 if ticks mod 5 = 0 [
   diffuse chemical (diffusion-rate / 100)
   ask simulation_area
   [
     if visual_object = 1 [ set chemical 0 stop ] ;;Don't drop pheromone on walls
     set chemical chemical * (100 - evaporation-rate) / 100  ;; slowly evaporate chemical
     recolor-patch
   ]
 ]
 check-boundaries
 update-fitness -0.005
 if ticks >= 10000 [
    compute-swarm-fitness
    let result_file "individual_result.csv"
    write_l2l_results result_file
    stop
 ]
 tick

end
@#$#@#$#@
GRAPHICS-WINDOW
154
23
970
520
-1
-1
8.0
1
10
1
1
1
0
1
1
1
0
100
0
60
0
0
1
ticks
30.0

PLOT
1137
87
1471
237
Membrane potential Neuron 1
Time
V
0.0
100.0
-80.0
-50.0
false
false
"" "if not enable_plots? [stop]\nif ticks > 0\n[\n   ;;set joblist lput synapsedelay joblist\n   if( length plot-list > 110 )\n   [\n     set plot-list remove-item 0 plot-list \n   ]\n   set plot-list lput ( [nmembranepotential] of one-of normalneurons with [nneuronlabel = neuronid_monitor1] ) plot-list\n   clear-plot\n   let xp 100\n   foreach plot-list [ [?]-> set xp xp + 1 plotxy xp - (length plot-list) ? ]\n   ;foreach plot-list [ [a[ ->  ( set xp xp + 1 plotxy xp - (length plot-list)) a  ]\n]"
PENS
"default" 1.0 0 -16777216 true "" ""

BUTTON
9
27
84
61
Setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
10
102
130
136
go one-step
go
NIL
1
T
OBSERVER
NIL
G
NIL
NIL
1

PLOT
1137
300
1472
450
Membrane potential Neuron 2
Time
V
0.0
100.0
-80.0
-50.0
false
false
"" "if not enable_plots? [stop]\nif ticks > 0\n[\n   ;;set joblist lput synapsedelay joblist\n   if( length plot-list2 > 110 )\n   [\n     set plot-list2 remove-item 0 plot-list2 \n   ]\n   set plot-list2 lput ( [nmembranepotential] of one-of normalneurons with [nneuronlabel = neuronid_monitor2] ) plot-list2\n   clear-plot\n   let xp 100\n   foreach plot-list2 [ [?]-> set xp xp + 1 plotxy xp - (length plot-list2) ? ]\n]"
PENS
"default" 1.0 0 -16777216 true "" ""

BUTTON
10
65
130
99
go-forever
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

INPUTBOX
1137
24
1249
84
neuronid_monitor1
302.0
1
0
Number

INPUTBOX
1137
239
1247
299
neuronid_monitor2
303.0
1
0
Number

PLOT
171
534
1125
724
Spike raster plot: each row represents the spike activity of a neuron. The lowest row corresponds to the neuron with the lowest neuron_id. Input (squared) neurons are not shown.
Time
Neuron
0.0
300.0
0.0
100.0
false
false
"" "if not enable_plots? [stop]\n   \nlet ypos 0\nif (ticks mod 300) = 0\n[\n  clear-plot\n]\nlet taxis (ticks mod 300)\nforeach ( sort-by [[?1 ?2]-> [nneuronlabel] of ?1 < [nneuronlabel] of ?2] normalneurons ) [;with [nlayernum = rasterized_layer] [\n  [?]->\n  ask ? [\n    set ypos ypos + 10;1\n    ifelse (color = red)\n    [\n      set-plot-pen-color black\n    ]\n    [\n      set-plot-pen-color 8\n    ]\n      plotxy taxis ypos\n      plotxy taxis (ypos + 1)\n      plotxy taxis (ypos + 2)\n      plotxy taxis (ypos + 3)\n      plotxy taxis (ypos + 4)\n      plotxy taxis (ypos + 5)\n      plotxy taxis (ypos + 6)\n      plotxy taxis (ypos + 7)\n  ]\n ]"
PENS
"default" 0.0 2 -16777216 true "" ""

SWITCH
5
238
142
271
awakecreature?
awakecreature?
0
1
-1000

SWITCH
6
587
129
620
noxious_red
noxious_red
0
1
-1000

SWITCH
6
621
129
654
noxious_white
noxious_white
0
1
-1000

SWITCH
6
656
129
689
noxious_green
noxious_green
1
1
-1000

SWITCH
5
278
144
311
istrainingmode?
istrainingmode?
0
1
-1000

BUTTON
10
178
130
214
re-draw World
draw-world
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
10
139
130
175
relocate insect
;ask patches with [ pxcor = 102 and pycor = 30 ] [set pcolor green]\nask testcreatures [setxy 82 30]
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
3
316
151
349
insect_view_distance
insect_view_distance
1
15
5.0
1
1
NIL
HORIZONTAL

SWITCH
3
457
152
490
enable_plots?
enable_plots?
1
1
-1000

SWITCH
3
358
151
391
leave_trail_on?
leave_trail_on?
1
1
-1000

SWITCH
3
398
152
431
show_sight_line?
show_sight_line?
1
1
-1000

PLOT
1144
534
1490
724
Efficacy of incoming synapses
NIL
NIL
0.0
5.0
0.0
10.0
true
false
"" "if not enable_plots? [stop]\nclear-plot \nset-histogram-num-bars 5\nset-plot-pen-color red\n\nlet cneu 0\nlet sumeff 0\nforeach  ( sort-by [ [?1 ?2]-> [presynneuronid] of ?1 < [presynneuronid] of ?2] normal-synapses with [possynneuronlabel = observed_efficacy_of_neuron] ) [\n  [?]->\n  ask ? [\n    set sumeff sumeff + synapseefficacy\n    plotxy cneu synapseefficacy\n    set cneu cneu + 1\n  ]\n]"
PENS
"default" 1.0 1 -16777216 true "" ""

INPUTBOX
1144
466
1300
526
observed_efficacy_of_neuron
303.0
1
0
Number

MONITOR
1300
21
1427
66
NIL
fitness_value
2
1
11

INPUTBOX
1796
78
1867
138
weight1
18.5
1
0
Number

INPUTBOX
1796
140
1871
200
weight2
15.0
1
0
Number

INPUTBOX
1796
202
1867
262
weight3
-11.5
1
0
Number

INPUTBOX
1796
264
1864
324
weight4
-11.5
1
0
Number

INPUTBOX
1796
326
1857
386
weight5
-15.5
1
0
Number

INPUTBOX
1796
389
1863
449
weight6
-15.0
1
0
Number

INPUTBOX
1796
452
1864
512
weight7
-11.0
1
0
Number

INPUTBOX
1796
515
1866
575
weight8
-9.5
1
0
Number

INPUTBOX
1796
577
1868
637
weight9
2.0
1
0
Number

INPUTBOX
1796
639
1868
699
weight10
-10.5
1
0
Number

INPUTBOX
1796
701
1861
761
weight11
10.0
1
0
Number

INPUTBOX
1796
763
1863
823
weight12
-8.0
1
0
Number

INPUTBOX
1796
826
1864
886
weight13
-20.0
1
0
Number

INPUTBOX
1797
889
1862
949
weight14
-11.0
1
0
Number

INPUTBOX
1797
951
1860
1011
weight15
-3.5
1
0
Number

INPUTBOX
1797
1014
1872
1074
weight16
2.5
1
0
Number

INPUTBOX
1798
1076
1867
1136
weight17
16.5
1
0
Number

INPUTBOX
1797
1139
1863
1199
weight18
5.5
1
0
Number

INPUTBOX
1797
1202
1868
1262
weight19
-9.0
1
0
Number

INPUTBOX
1797
1264
1870
1324
weight20
20.0
1
0
Number

INPUTBOX
1797
1326
1865
1386
weight21
6.0
1
0
Number

INPUTBOX
1797
1388
1863
1448
weight22
-13.5
1
0
Number

INPUTBOX
1797
1451
1863
1511
weight23
8.0
1
0
Number

INPUTBOX
1797
1514
1864
1574
weight24
20.0
1
0
Number

INPUTBOX
1797
1576
1862
1636
weight25
20.0
1
0
Number

INPUTBOX
1797
1639
1865
1699
weight26
10.0
1
0
Number

INPUTBOX
1797
1702
1864
1762
weight27
5.0
1
0
Number

INPUTBOX
1797
1765
1862
1825
weight28
-1.0
1
0
Number

INPUTBOX
1797
1828
1861
1888
weight29
-0.5
1
0
Number

INPUTBOX
1797
1891
1863
1951
weight30
15.5
1
0
Number

INPUTBOX
1797
1954
1867
2014
weight31
-12.0
1
0
Number

INPUTBOX
1798
2017
1864
2077
weight32
18.0
1
0
Number

INPUTBOX
1798
2080
1864
2140
weight33
-7.5
1
0
Number

INPUTBOX
1798
2143
1860
2203
weight34
19.5
1
0
Number

INPUTBOX
1798
2206
1861
2266
weight35
-14.5
1
0
Number

INPUTBOX
1798
2268
1863
2328
weight36
-12.5
1
0
Number

INPUTBOX
1798
2331
1875
2391
weight37
18.0
1
0
Number

INPUTBOX
1798
2394
1869
2454
weight38
8.0
1
0
Number

INPUTBOX
1798
2456
1861
2516
weight39
-6.0
1
0
Number

INPUTBOX
1798
2519
1861
2579
weight40
-13.0
1
0
Number

INPUTBOX
1798
2582
1861
2642
weight41
-15.0
1
0
Number

INPUTBOX
1798
2644
1858
2704
weight42
-14.0
1
0
Number

INPUTBOX
1798
2706
1866
2766
weight43
-1.5
1
0
Number

INPUTBOX
1799
2769
1871
2829
weight44
5.0
1
0
Number

INPUTBOX
1799
2832
1870
2892
weight45
14.0
1
0
Number

INPUTBOX
1799
2895
1867
2955
weight46
5.0
1
0
Number

INPUTBOX
1799
2958
1864
3018
weight47
-1.0
1
0
Number

INPUTBOX
1799
3020
1861
3080
weight48
-19.0
1
0
Number

INPUTBOX
1799
3082
1866
3142
weight49
17.0
1
0
Number

INPUTBOX
1799
3145
1865
3205
weight50
6.0
1
0
Number

INPUTBOX
1799
3208
1865
3268
weight51
-18.5
1
0
Number

INPUTBOX
1799
3270
1867
3330
weight52
18.5
1
0
Number

INPUTBOX
1800
3333
1866
3393
weight53
-20.0
1
0
Number

INPUTBOX
1800
3395
1867
3455
weight54
20.0
1
0
Number

INPUTBOX
1800
3458
1866
3518
weight55
-11.0
1
0
Number

INPUTBOX
1799
3522
1859
3582
weight56
-10.0
1
0
Number

INPUTBOX
1798
3586
1862
3646
weight57
9.5
1
0
Number

INPUTBOX
1799
3649
1861
3709
weight58
0.5
1
0
Number

INPUTBOX
1799
3711
1862
3771
weight59
13.5
1
0
Number

INPUTBOX
1799
3773
1863
3833
weight60
-18.5
1
0
Number

INPUTBOX
1799
3835
1864
3895
weight61
3.5
1
0
Number

INPUTBOX
1799
3897
1861
3957
weight62
18.0
1
0
Number

INPUTBOX
1798
3959
1861
4019
weight63
-18.5
1
0
Number

INPUTBOX
1799
4021
1860
4081
weight64
19.0
1
0
Number

INPUTBOX
1799
4083
1865
4143
weight65
-13.0
1
0
Number

INPUTBOX
1799
4145
1864
4205
weight66
-20.0
1
0
Number

INPUTBOX
1799
4207
1866
4267
weight67
15.5
1
0
Number

INPUTBOX
1799
4269
1864
4329
weight68
-9.0
1
0
Number

INPUTBOX
1799
4331
1860
4391
weight69
5.0
1
0
Number

INPUTBOX
1799
4393
1861
4453
weight70
-15.0
1
0
Number

INPUTBOX
1799
4455
1868
4515
weight71
-6.0
1
0
Number

INPUTBOX
1799
4518
1868
4578
weight72
17.5
1
0
Number

INPUTBOX
1799
4580
1864
4640
weight73
-8.0
1
0
Number

INPUTBOX
1799
4643
1863
4703
weight74
-10.0
1
0
Number

INPUTBOX
1799
4705
1864
4765
weight75
-1.5
1
0
Number

INPUTBOX
1799
4768
1868
4828
weight76
10.5
1
0
Number

INPUTBOX
1799
4831
1868
4891
weight77
11.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight78
9.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight79
14.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight80
-10.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight81
-7.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight82
3.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight83
11.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight84
5.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight85
16.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight86
-4.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight87
-16.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight88
-4.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight89
0.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight90
12.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight91
-14.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight92
17.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight93
-1.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight94
-1.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight95
16.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight96
-13.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight97
6.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight98
1.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight99
-14.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight100
6.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight101
10.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight102
-4.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight103
-19.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight104
-4.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight105
-5.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight106
-5.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight107
-5.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight108
10.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight109
4.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight110
11.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight111
17.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight112
-3.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight113
16.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight114
-6.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight115
-10.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight116
-19.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight117
-3.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight118
-7.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight119
4.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight120
17.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight121
-19.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight122
13.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight123
18.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight124
-18.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight125
16.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight126
-11.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight127
-16.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight128
-2.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight129
-7.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight130
20.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight131
-20.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight132
8.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight133
-10.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight134
18.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight135
-14.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight136
15.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight137
5.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight138
-10.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight139
-8.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight140
-4.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight141
-3.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight142
3.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight143
-8.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight144
1.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight145
7.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight146
-16.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight147
12.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight148
6.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight149
-0.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight150
-10.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight151
7.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight152
-9.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight153
3.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight154
11.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight155
20.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight156
15.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight157
-6.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight158
6.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight159
7.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight160
-13.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight161
17.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight162
11.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight163
-5.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight164
8.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight165
-4.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight166
-4.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight167
-5.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight168
-7.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight169
-11.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight170
-8.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight171
12.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight172
-9.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight173
18.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight174
-7.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight175
-15.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight176
-20.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight177
-17.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight178
20.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight179
-11.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight180
-4.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight181
-11.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight182
-8.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight183
4.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight184
-9.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight185
-17.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight186
1.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight187
9.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight188
-11.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight189
12.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight190
-11.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight191
-7.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight192
-16.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight193
-7.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight194
1.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight195
11.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight196
-8.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight197
6.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight198
3.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight199
5.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight200
8.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight201
12.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight202
-14.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight203
-10.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight204
20.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight205
-19.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight206
-20.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight207
-8.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight208
5.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight209
12.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight210
-10.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight211
11.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight212
13.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight213
12.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight214
9.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight215
-14.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight216
-20.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight217
-2.0
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight218
15.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight219
19.5
1
0
Number

INPUTBOX
1800
4895
1874
4955
weight220
2.0
1
0
Number

INPUTBOX
1893
77
1955
137
delay1
7.0
1
0
Number

INPUTBOX
1893
141
1955
201
delay2
3.0
1
0
Number

INPUTBOX
1893
205
1955
265
delay3
3.0
1
0
Number

INPUTBOX
1893
269
1955
329
delay4
7.0
1
0
Number

INPUTBOX
1893
334
1955
394
delay5
7.0
1
0
Number

INPUTBOX
1893
398
1955
458
delay6
2.0
1
0
Number

INPUTBOX
1893
461
1955
521
delay7
2.0
1
0
Number

INPUTBOX
1893
525
1955
585
delay8
7.0
1
0
Number

INPUTBOX
1893
588
1955
648
delay9
7.0
1
0
Number

INPUTBOX
1893
652
1955
712
delay10
5.0
1
0
Number

INPUTBOX
1893
715
1955
775
delay11
7.0
1
0
Number

INPUTBOX
1893
778
1955
838
delay12
7.0
1
0
Number

INPUTBOX
1893
841
1955
901
delay13
2.0
1
0
Number

INPUTBOX
1893
904
1955
964
delay14
4.0
1
0
Number

INPUTBOX
1893
967
1955
1027
delay15
2.0
1
0
Number

INPUTBOX
1893
1029
1955
1089
delay16
4.0
1
0
Number

INPUTBOX
1893
1093
1955
1153
delay17
7.0
1
0
Number

INPUTBOX
1893
1156
1955
1216
delay18
5.0
1
0
Number

INPUTBOX
1893
1219
1955
1279
delay19
6.0
1
0
Number

INPUTBOX
1893
1282
1955
1342
delay20
3.0
1
0
Number

INPUTBOX
1893
1346
1955
1406
delay21
5.0
1
0
Number

INPUTBOX
1893
1410
1955
1470
delay22
1.0
1
0
Number

INPUTBOX
1893
1474
1955
1534
delay23
2.0
1
0
Number

INPUTBOX
1893
1538
1955
1598
delay24
7.0
1
0
Number

INPUTBOX
1893
1602
1955
1662
delay25
5.0
1
0
Number

INPUTBOX
1893
1665
1955
1725
delay26
3.0
1
0
Number

INPUTBOX
1893
1729
1955
1789
delay27
1.0
1
0
Number

INPUTBOX
1893
1792
1955
1852
delay28
1.0
1
0
Number

INPUTBOX
1893
1856
1955
1916
delay29
4.0
1
0
Number

INPUTBOX
1893
1920
1955
1980
delay30
2.0
1
0
Number

INPUTBOX
1893
1983
1955
2043
delay31
7.0
1
0
Number

INPUTBOX
1893
2045
1955
2105
delay32
2.0
1
0
Number

INPUTBOX
1893
2108
1955
2168
delay33
6.0
1
0
Number

INPUTBOX
1893
2171
1955
2231
delay34
6.0
1
0
Number

INPUTBOX
1893
2236
1955
2296
delay35
7.0
1
0
Number

INPUTBOX
1893
2300
1955
2360
delay36
6.0
1
0
Number

INPUTBOX
1893
2365
1955
2425
delay37
2.0
1
0
Number

INPUTBOX
1893
2429
1955
2489
delay38
7.0
1
0
Number

INPUTBOX
1893
2493
1955
2553
delay39
3.0
1
0
Number

INPUTBOX
1893
2557
1955
2617
delay40
1.0
1
0
Number

INPUTBOX
1893
2620
1955
2680
delay41
7.0
1
0
Number

INPUTBOX
1893
2684
1955
2744
delay42
7.0
1
0
Number

INPUTBOX
1893
2748
1955
2808
delay43
4.0
1
0
Number

INPUTBOX
1893
2812
1955
2872
delay44
5.0
1
0
Number

INPUTBOX
1893
2876
1955
2936
delay45
5.0
1
0
Number

INPUTBOX
1893
2941
1955
3001
delay46
4.0
1
0
Number

INPUTBOX
1893
3005
1955
3065
delay47
4.0
1
0
Number

INPUTBOX
1893
3067
1955
3127
delay48
4.0
1
0
Number

INPUTBOX
1893
3129
1955
3189
delay49
5.0
1
0
Number

INPUTBOX
1893
3192
1955
3252
delay50
1.0
1
0
Number

INPUTBOX
1893
3255
1955
3315
delay51
5.0
1
0
Number

INPUTBOX
1893
3322
1955
3382
delay52
1.0
1
0
Number

INPUTBOX
1893
3386
1955
3446
delay53
2.0
1
0
Number

INPUTBOX
1893
3452
1955
3512
delay54
7.0
1
0
Number

INPUTBOX
1893
3516
1955
3576
delay55
2.0
1
0
Number

INPUTBOX
1893
3581
1955
3641
delay56
1.0
1
0
Number

INPUTBOX
1893
3646
1955
3706
delay57
1.0
1
0
Number

INPUTBOX
1893
3709
1955
3769
delay58
3.0
1
0
Number

INPUTBOX
1893
3773
1955
3833
delay59
4.0
1
0
Number

INPUTBOX
1893
3837
1955
3897
delay60
3.0
1
0
Number

INPUTBOX
1893
3901
1955
3961
delay61
3.0
1
0
Number

INPUTBOX
1893
3965
1955
4025
delay62
5.0
1
0
Number

INPUTBOX
1893
4031
1955
4091
delay63
4.0
1
0
Number

INPUTBOX
1893
4096
1955
4156
delay64
6.0
1
0
Number

INPUTBOX
1893
4160
1955
4220
delay65
5.0
1
0
Number

INPUTBOX
1893
4225
1955
4285
delay66
3.0
1
0
Number

INPUTBOX
1893
4289
1955
4349
delay67
3.0
1
0
Number

INPUTBOX
1893
4354
1955
4414
delay68
3.0
1
0
Number

INPUTBOX
1893
4418
1955
4478
delay69
5.0
1
0
Number

INPUTBOX
1893
4482
1955
4542
delay70
5.0
1
0
Number

INPUTBOX
1893
4548
1955
4608
delay71
2.0
1
0
Number

INPUTBOX
1893
4615
1955
4675
delay72
7.0
1
0
Number

INPUTBOX
1893
4680
1955
4740
delay73
2.0
1
0
Number

INPUTBOX
1893
4745
1955
4805
delay74
2.0
1
0
Number

INPUTBOX
1893
4810
1955
4870
delay75
2.0
1
0
Number

INPUTBOX
1893
4875
1955
4935
delay76
4.0
1
0
Number

INPUTBOX
1893
4942
1955
5002
delay77
3.0
1
0
Number

INPUTBOX
1893
5008
1955
5068
delay78
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay79
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay80
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay81
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay82
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay83
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay84
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay85
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay86
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay87
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay88
4.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay89
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay90
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay91
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay92
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay93
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay94
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay95
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay96
4.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay97
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay98
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay99
4.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay100
4.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay101
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay102
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay103
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay104
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay105
4.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay106
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay107
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay108
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay109
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay110
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay111
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay112
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay113
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay114
4.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay115
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay116
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay117
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay118
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay119
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay120
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay121
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay122
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay123
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay124
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay125
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay126
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay127
4.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay128
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay129
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay130
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay131
4.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay132
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay133
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay134
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay135
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay136
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay137
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay138
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay139
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay140
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay141
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay142
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay143
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay144
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay145
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay146
4.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay147
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay148
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay149
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay150
4.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay151
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay152
4.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay153
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay154
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay155
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay156
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay157
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay158
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay159
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay160
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay161
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay162
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay163
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay164
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay165
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay166
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay167
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay168
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay169
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay170
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay171
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay172
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay173
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay174
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay175
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay176
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay177
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay178
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay179
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay180
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay181
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay182
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay183
4.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay184
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay185
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay186
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay187
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay188
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay189
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay190
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay191
4.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay192
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay193
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay194
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay195
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay196
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay197
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay198
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay199
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay200
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay201
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay202
7.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay203
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay204
4.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay205
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay206
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay207
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay208
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay209
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay210
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay211
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay212
6.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay213
4.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay214
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay215
2.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay216
5.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay217
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay218
1.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay219
3.0
1
0
Number

INPUTBOX
1893
4895
1955
4955
delay220
5.0
1
0
Number

SLIDER
3
500
150
533
evaporation-rate
evaporation-rate
0
20
2.0
1
1
NIL
HORIZONTAL

SLIDER
3
534
150
567
diffusion-rate
diffusion-rate
0.0
99.0
2.0
1.0
1
NIL
HORIZONTAL

INPUTBOX
991
51
1087
111
inject_neuron
300.0
1
0
Number

BUTTON
990
116
1094
149
inject neuron
ask normalneurons with [nneuronlabel = inject_neuron]\n[\n  ;receive-pulse [ #prneuronid #snefficacy #excinh #plearningon? ];\n   receive-pulse 1 10 excitatory_synapse false\n]
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

@#$#@#$#@
## WHAT IS IT?

The presented Spiking Neural Network (SNN) model is built in the framework of generalized Integrate-and-fire models which recreate to some extend the phenomenological dynamics of neurons while abstracting the biophysical processes behind them. The Spike Timing Dependent Plasticity (STDP) learning approach proposed by (Gerstner and al. 1996, Kempter et al. 1999)  has been implemented and used as the underlying learning mechanism for the experimental neural circuit.

The neural circuit implemented in this model enables a simulated agent representing a virtual insect to move in a two dimensional world, learning to visually identify and avoid noxious stimuli while moving towards perceived rewarding stimuli. At the beginning, the agent is not aware of which stimuli are to be avoided or followed. Learning occurs through  Reward-and-punishment classical conditioning. Here the agent learns to associate different colours with unconditioned reflex responses.

## HOW IT WORKS

The experimental virtual-insect is able to process three types of sensorial information: (1) visual, (2) pain and (3) pleasant or rewarding sensation.  The visual information  is acquired through three photoreceptors where each one of them is sensitive to one specific color (white, red or green). Each photoreceptor is connected with one afferent neuron which propagates the input pulses towards two Motoneurons identified by the labels 21 and 22. Pain is elicited by a nociceptor (labeled 4) whenever the insect collides with a wall or a noxious stimulus. A rewarding or pleasant sensation is elicited by a pheromone (or nutrient smell) sensor (labeled 5) when the insect gets in direct contact with the originating stimulus.

The motor system allows the virtual insect to move forward (neuron labeled 31) or rotate in one direction (neuron labeled 32) according to the reflexive behaviour associated to it. In order to keep the insect moving even in the absence of external stimuli, the motoneuron 22 is connected to a neural oscillator sub-circuit composed of two neurons (identified by the labels 23 and 24)  performing the function of a pacemaker which sends a periodic pulse to Motoneuron 22. The pacemaker is initiated by a pulse from an input neuron (labeled 6) which represents an external input current (i.e; intracellular electrode).

## HOW TO USE IT

1. Press Setup to create:
 a. the neural circuit (on the left of the view)
 b. the insect and its virtual world (on the right of the view)

2. Press go-forever to continually run the simulation.

3. Press re-draw world to change the virtual world by adding random patches.

4. Press relocate insect to bring the insect to its initial (center) position.

5. Use the awake creature switch to enable or disable the movement of the insect.

6. Use the colours switches to indicate which colours are associated to harmful stimuli.

7. Use the insect_view_distance slider to indicate the number of patches the insect can
   look ahead.

8. Use the leave_trail_on? switch to follow the movement of the insect.


On the circuit side:
Input neurons are depicted with green squares.
Normal neurons are represented with pink circles. When a neuron fires its colour changes to red for a short time (for 1 tick or iteration).
Synapses are represented by links (grey lines) between neurons. Inhibitory synapses are depicted by red lines.

If istrainingmode? is on then the training phase is active. During the training phase, the insect is moved one patch forward everytime it is on a patch associated with a noxious stimulus. Otherwise, the insect would keep rotating over the noxious patch. Also, the insect is repositioned in its initial coordinates every time it reaches the virtual-world boundaries.


## THINGS TO NOTICE

At the beginning the insect moves along the virtual-world in a seemingly random way colliding equally with all types of coloured patches. This demonstrates the initial inability of the insect to discriminate and react in response to visual stimuli.  However, after a few thousands iterations (depending on the learning parameters), it can be seen that the trajectories lengthen as the learning of the insect progresses and more obstacles (walls and harmful stimuli) are avoided.

## THINGS TO TRY

Follow the dynamic of single neurons by monitoring the membrane potential plots while running the simulation step by step.

Set different view distances to see if the behaviour of the insect changes.

Manipulate the STDP learning parameters. Which parameters speed up or slow down the adaptation to the environment?

## EXTENDING THE MODEL

Use different kernels for the decay() and epsp() functions to make the model more accurate in biological terms.

## NETLOGO FEATURES

Use of link and list primitives.


## CREDITS AND REFERENCES

If you mention this model in a publication, we ask that you include these citations for the model itself and for the NetLogo software:


* Cristian Jimenez-Romero, David Sousa-Rodrigues, Jeffrey H. Johnson, Vitorino Ramos
 A Model for Foraging Ants, Controlled by Spiking Neural Networks and Double Pheromonesin UK Workshop on Computational Intelligence 2015, University of Exeter, September 2015.


* Wilensky, U. (1999). NetLogo. http://ccl.northwestern.edu/netlogo/. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.2.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="experiment1" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>count turtles</metric>
  </experiment>
</experiments>
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
