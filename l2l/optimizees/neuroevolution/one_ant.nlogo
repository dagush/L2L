;; Neuro-evolution with SpikingLab: Modelling Agents with Brains
;; Author: Cristian Jimenez Romero - FZJ - 2021

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Include SpikingLab SNN related code:;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
__includes [ "spikinglab.nls" ]

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Artificial Ant related code:;;;;;;;;;;;;;;;;;;;;;;;
breed [testcreatures testcreature]
breed [visualsensors visualsensor]

breed [sightlines sightline]
directed-link-breed [sight-trajectories sight-trajectory]

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
breed [itrails itrail]
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;
;;; Global variables
;;;
globals [

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;World globals;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  min_x_boundary ;;Define initial x coordinate of simulation area
  min_y_boundary ;;Define initial y coordinate of simulation area
  max_x_boundary ;;Define final x coordinate of simulation area
  max_y_boundary ;;Define final y coordinate of simulation area
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;GA Globals;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  fitness_value
  raw_weight_data
  raw_delay_data

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Count Collisions;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  collisions

]

testcreatures-own [

  creature_label
  creature_id
  reward_neuron
  pain_neuron
  move_neuron
  rotate_neuron
  creature_sightline

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

;;;
;;; Set global variables with their initial values
;;;
to initialize-global-vars

  initialize-spikinglab-globals

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;Simulation area globals;;;;;;;;;;;;;;;;;;;;;;;;
  set min_x_boundary 80  ;;Define initial x coordinate of simulation area
  set min_y_boundary 1   ;;Define initial y coordinate of simulation area
  set max_x_boundary 120 ;;Define final x coordinate of simulation area
  set max_y_boundary 60  ;;Define final y coordinate of simulation area

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Insect globals;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  set pspikefrequency 1
  set error_free_counter 0
  set required_error_free_iterations 35000

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Genetic Algorithm ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  set fitness_value 0

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Collisions;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  set collisions 0

end

;;;
;;; Create insect agent
;;;
to-report create-creature [#xpos #ypos #creature_label #reward_neuron_label #pain_neuron_label #move_neuron_label #rotate_neuron_label]

  let reward_neuron_id get-input-neuronid-from-label #reward_neuron_label
  let pain_neuron_id get-input-neuronid-from-label #pain_neuron_label
  let move_neuron_id get-neuronid-from-label #move_neuron_label
  let rotate_neuron_id get-neuronid-from-label #rotate_neuron_label
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
    set rotate_neuron rotate_neuron_id
    set creature_id who
    set returned_id creature_id

  ]
  report  returned_id

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
;;; Ask photoreceptor if there is a patch ahead (within insect_view_distance) with a perceivable colour (= attached_to_colour)
;;;
to view-world-ahead ;;Called by visualsensors

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
  while [itemcount <= view_distance and foundobj = black] ;;Project a sight line up to view_distance number of patches
  [
    set itemcount itemcount + 0.34 ;Increase in steps of 0.3 patches to check if there is an object (reward or harming stimulus) in between
    ask patch-ahead itemcount [
       set foundobj pcolor
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
 ask patch-here [ set onobject pcolor ]
 ifelse (onobject = white)
 [
    ifelse (noxious_white) ;;is White attached to a noxious stimulus
    [
      feed-input-neuron pain_neuron 1 ;;induce Pain
      update-fitness -1.5
      update-collision-counter
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
        update-fitness -1.5
        update-collision-counter
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
           update-fitness -1.5
           update-collision-counter
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

 let dorotation? false
 let domovement? false
 ;;Check rotate actuator
 ask normalneuron rotate_neuron [
   if (nlast-firing-time = ticks);
   [
     set dorotation? true
   ]
 ]
 ;;Check move forward actuator
 ask normalneuron move_neuron[
   if (nlast-firing-time = ticks);
   [
     set domovement? true
   ]
 ]

 if (dorotation?)
 [
   rotate-creature 4
 ]

 if (domovement?)
 [
   move-creature 0.4
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


to read_l2l_config [ #filename ]
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

to load_test_brain_data
  set weight1    -11.2
  set weight2    -14.2
  set weight3      -20
  set weight4       20
  set weight5  -2.80000
  set weight6  19.8000
  set weight7    -16.6
  set weight8  8.20000
  set weight9  1.60000
  set weight10      20
  set weight11 3.80000
  set weight12   -16.4
  set weight13      20
  set weight14     -20
  set weight15    -5.6
  set weight16   -16.2
  set weight17     -14
  set weight18 -5.80000
  set weight19   -18.8
  set weight20 13.6000
  set weight21      20
  set weight22    -6.6
  set weight23 -13.4000
  set weight24    -5.6
  set weight25 -8.80000
  set weight26     -13
  set weight27 6.80000
  set weight28 12.6000
  set weight29      -8
  set weight30     -14
  set weight31 13.2000
  set weight32      -9
  set weight33   -14.6
  set weight34       6
  set weight35 5.20000
  set weight36 -2.40000
  set weight37   -14.2
  set weight38 5.60000
  set weight39   -18.2
  set weight40 -4.40000
  set weight41   -14.2
  set weight42     -19
  set weight43 7.80000
  set weight44 4.60000
  set weight45    19.6
  set weight46 19.2000
  set weight47 -6.80000
  set weight48     -20
  set weight49      20
  set weight50 -0.800000
  set weight51 -3.60000
  set weight52   -16.2
  set weight53 15.2000
  set weight54   -19.2
  set weight55     -20
  set weight56    -5.6
  set weight57     -17
  set weight58      13
  set weight59     -20
  set weight60 -2.80000
  set weight61 16.2000
  set weight62   -18.6
  set weight63   -16.2
  set weight64     -20
  set weight65 -10.8000
  set weight66 0.400000
  set weight67   -15.2
  set weight68 18.4000
  set weight69      20
  set weight70   -10.2
  set weight71 9.20000
  set weight72   -11.4
  set weight73      20
  set weight74   -18.2
  set weight75 4.60000
  set weight76 -5.20000
  set weight77 13.6000
  set weight78 -13.4000
  set delay1         3
  set delay2         3
  set delay3         6
  set delay4         7
  set delay5         2
  set delay6         3
  set delay7         4
  set delay8         5
  set delay9         7
  set delay10        7
  set delay11        1
  set delay12        3
  set delay13        3
  set delay14        3
  set delay15        2
  set delay16        4
  set delay17        3
  set delay18        1
  set delay19        6
  set delay20        5
  set delay21        6
  set delay22        6
  set delay23        7
  set delay24        3
  set delay25        5
  set delay26        1
  set delay27        6
  set delay28        7
  set delay29        4
  set delay30        1
  set delay31        2
  set delay32        3
  set delay33        2
  set delay34        5
  set delay35        3
  set delay36        7
  set delay37        4
  set delay38        2
  set delay39        7
  set delay40        4
  set delay41        6
  set delay42        1
  set delay43        4
  set delay44        6
  set delay45        5
  set delay46        1
  set delay47        4
  set delay48        3
  set delay49        2
  set delay50        4
  set delay51        3
  set delay52        1
  set delay53        2
  set delay54        1
  set delay55        6
  set delay56        6
  set delay57        7
  set delay58        7
  set delay59        4
  set delay60        1
  set delay61        2
  set delay62        3
  set delay63        6
  set delay64        3
  set delay65        4
  set delay66        1
  set delay67        1
  set delay68        3
  set delay69        6
  set delay70        6
  set delay71        4
  set delay72        1
  set delay73        5
  set delay74        2
  set delay75        3
  set delay76        1
  set delay77        4
  set delay78        2
end

to write_l2l_results [ #filename ]
  file-close-all
  file-open #filename
  file-print raw_weight_data
  file-print raw_delay_data
  file-print fitness_value
  file-flush
  file-close-all

end


to load_l2l_parameters
  let weight_data[]
  let plasticity_data[]
  let delay_data[]
  let current_index 1

  ;;;;;;; Get data from the weight and delay input boxes in the Graphical User Interface (GUI).
  ;;;;;;; The BehaviourSearch tool uses the input boxes in the GUI to set the parameters of the simulation
  repeat 78 [ ;;The Spiking Neural Network (SNN) contains 78 connections (synapses) in total
    set weight_data lput (runresult (word "weight" current_index)) weight_data
    ;;Deactivate plasticity in this experiment, use 0 instead of: set plasticity_data lput (runresult (word "plasticity" current_index)) plasticity_data
    set plasticity_data lput 0 plasticity_data
    set delay_data lput (runresult (word "delay" current_index)) delay_data
    set current_index current_index + 1
  ]
  setup_neural_parameters

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Call procedure below to test the optimized parameters:
  ;load_test_brain_data
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  let n_insects 1
  let insect_offset 0
  repeat n_insects [
    build_neural_circuit insect_offset weight_data plasticity_data delay_data
    set insect_offset insect_offset + 1000
  ]

end

to setup_neural_parameters

  ;;;;;;;;;;;;;;Setup Neuron types;;;;;;;;;;;;;;
  ;;;;;;;;;;;;;;;;Neuron type1:
  setup-neurontype 1 -65 -55 0.08 -75 3 -75 ;(typeid restpot threshold decayr refractpot refracttime)
  set-neurontype-learning-params 1 0.045 75 0.045 -50 15 1 (8 + 3) (15 + 3) ;[ #pneurontypeid #ppos_hebb_weight #ppos_time_window #pneg_hebb_weight #pneg_time_window #pmax_synaptic_weight
                                    ;#pmin_synaptic_weight #ppos_syn_change_interval #pneg_syn_change_interval]


  ;;;;;;;;;;;;;;;;Neuron type2:
  setup-neurontype 2 -65 -55 0.08 -70 1 -70 ;(typeid restpot threshold decayr refractpot refracttime)
  set-neurontype-learning-params 2 0.09 55 0.09 -25 15 1 8 15 ;[ #pneurontypeid #ppos_hebb_weight #ppos_time_window #pneg_hebb_weight #pneg_time_window #pmax_synaptic_weight
                                    ;#pmin_synaptic_weight #ppos_syn_change_interval #pneg_syn_change_interval]

end

to build_neural_circuit [ #insect_offset #weight_data #plasticity_data #delay_data ]


  ;;;;;;;;;;;;;;Layer 1: Visual Afferent neurons with receptors ;;;;;;;;;;;;;;;
  setup-normal-neuron 1 15 10  (#insect_offset + 11) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron 5 10  (#insect_offset + 1)  (#insect_offset + 11) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron psynapseefficacy pcoding pnumofspikes]

  setup-normal-neuron 1 15 15  (#insect_offset + 12) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron    5 15  (#insect_offset + 2)  (#insect_offset + 12) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron ipsynapseefficacy pcoding pnumofspikes]

  setup-normal-neuron 1 15 20  (#insect_offset + 13) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron    5 20  (#insect_offset + 3)  (#insect_offset + 13) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron ipsynapseefficacy pcoding pnumofspikes]

  ;;;;;;;;;;;;;;; Layer 1b: nociceptive and reward neurons with receptors ;;;;;;;;;;;;;;;;;;;;;;
  ;Nociceptive pathway:
  setup-normal-neuron 1 15 25  (#insect_offset + 14) 1
  setup-input-neuron   5 25  (#insect_offset + 4)  (#insect_offset + 14) 20 1 pspikefrequency

  ;Reward pathway:
  setup-normal-neuron 1 15 30  (#insect_offset + 15) 1
  setup-input-neuron   5 30  (#insect_offset + 5)  (#insect_offset + 15) 20 1 pspikefrequency


  ;;;;;;Create heart layer 1c:
  setup-normal-neuron 2 15 40  (#insect_offset + 16) 2
  setup-normal-neuron 2 15 50  (#insect_offset + 17) 2
  setup-synapse  (#insect_offset + 16)  (#insect_offset + 17) 15 excitatory_synapse 2 false ;(no plasticity needed)
  setup-synapse  (#insect_offset + 17)  (#insect_offset + 16) 15 excitatory_synapse 3 false ;(no plasticity needed)
  setup-input-neuron 10 40  (#insect_offset + 6)  (#insect_offset + 16) 20 1 pspikefrequency ;;Voltage clamp to start pacemaker


  ;;;;;;Create neuronal layer 2 (6):
  let neuron_counter 0
  ;; x = r cos(t) y = r sin(t)
  repeat 6 [
    setup-normal-neuron 3 (10 * cos(neuron_counter * 60) + 40) ( 10 * sin(neuron_counter * 60) + 20)  (#insect_offset + 20 + neuron_counter) 1 ;;[layernum pposx pposy pid pneurontypeid]
    set neuron_counter neuron_counter + 1
  ]


  ;;;;;;Create neuronal motor layer 3 (2):
  set neuron_counter 0
  repeat 2 [
    setup-normal-neuron 4 65 ( 17 + neuron_counter * 5)  (#insect_offset + 30 + neuron_counter) 1 ;;[layernum pposx pposy pid pneurontypeid]
    set neuron_counter neuron_counter + 1
  ]


  ;;;;;;;;;;;;;;;;;;;;;;Create Creature and attach neural circuit
  ;; Start insect hearth
  feed-input-neuron_by_label  (#insect_offset + 6) 1

  let creatureid create-creature 102 30 (#insect_offset + 1)  (#insect_offset + 5)  (#insect_offset + 4)  (#insect_offset + 31)  (#insect_offset + 30);;[#xpos #ypos #creature_id #reward_input_neuron #pain_input_neuron #move_neuron #rotate_neuron]
  ;;;;;;;;;;Create Visual sensors and attach neurons to them;;;;;;;;;
  create-visual-sensor  (#insect_offset + 1) 0  white  (#insect_offset + 1) creatureid;[ psensor_id pposition colour_sensitive attached_neuron attached_creature]
  create-visual-sensor  (#insect_offset + 2) 0 red (#insect_offset + 2) creatureid;[ psensor_id pposition colour_sensitive attached_neuron attached_creature]
  create-visual-sensor  (#insect_offset + 3) 0 green (#insect_offset + 3) creatureid;[ psensor_id pposition colour_sensitive attached_neuron attached_creature]

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
  repeat 6 [
    set layer_to 0
    repeat 6 [
      let current_weight item current_index #weight_data
      let current_delay item current_index #delay_data
      let current_plasticity item current_index #plasticity_data
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

      setup-synapse  (#insect_offset + 11 + layer_from)  (#insect_offset + 20 + layer_to) abs(current_weight) type_exc_or_inh_synapse current_delay plasticity?
      set current_index current_index + 1
      set layer_to layer_to + 1
    ]
    set layer_from layer_from + 1
  ]

  ;;;;;;;;;;;;;;;;;;;;Create synapses from layer 2 to layer 2:
  set layer_from 0
  set layer_to 0
  repeat 6 [
    set layer_to 0
    repeat 6 [
      let current_weight item current_index #weight_data
      let current_delay item current_index #delay_data
      let current_plasticity item current_index #plasticity_data
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
        setup-synapse  (#insect_offset + 20 + layer_from)  (#insect_offset + 20 + layer_to) abs(current_weight) type_exc_or_inh_synapse current_delay plasticity?
        set current_index current_index + 1
      ]
      set layer_to layer_to + 1
    ]
    set layer_from layer_from + 1
  ]

  ;;;;;;;;;;;;;;;;;;;;Create synapses from layer 2 to layer 3:
  set layer_from 0
  set layer_to 0
  repeat 6 [
    set layer_to 0
    repeat 2 [
      let current_weight item current_index #weight_data
      let current_delay item current_index #delay_data
      let current_plasticity item current_index #plasticity_data
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

      setup-synapse  (#insect_offset + 20 + layer_from)  (#insect_offset + 30 + layer_to) abs(current_weight) type_exc_or_inh_synapse current_delay plasticity?
      set current_index current_index + 1
      set layer_to layer_to + 1
    ]
    set layer_from layer_from + 1
  ]

end

to update-fitness [ #fitness_change ]
  set fitness_value fitness_value + #fitness_change
end

to update-collision-counter
  set collisions collisions + 1
end

;;;
;;; Create neural circuit, insect and world
;;;
to setup

  clear-all

  RESET-TICKS

  initialize-global-vars
  random-seed 7850
  ;;;;;;;;;;Draw world with white, green and red patches;;;;;;;;
  draw-world
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  load_l2l_parameters

end


;;;
;;; Generate insect world with 3 types of patches
;;;
to draw-world

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
       ;set istrainingmode? false
      ]
      ask testcreatures [
        if (xcor < 80 or xcor > 120) or (ycor < 1 or ycor > 60)
        [
          setxy 102 30
        ]
      ]
   ]

end

;;;
;;; Run simulation
;;;
to go

 if (awakecreature?)
 [
   ask itrails [ check-trail ]
   ask visualsensors [ view-world-ahead ] ;;Feed visual sensors at first
   ask testcreatures [ perceive-world]; integrate visual and touch information
   do-network-dynamics
   ask testcreatures [do-actuators]
 ]

 check-boundaries
 tick

 if ticks >= ( 500000 - 1)
 [
    let result_file "individual_result.csv"
    write_l2l_results result_file
    stop
 ]

end
@#$#@#$#@
GRAPHICS-WINDOW
147
10
1073
481
-1
-1
7.59
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
120
0
60
0
0
1
ticks
30.0

PLOT
373
669
727
859
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
0
10
75
44
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
0
78
120
112
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
0
669
373
859
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
0
44
120
78
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
1073
10
1198
70
neuronid_monitor1
40.0
1
0
Number

INPUTBOX
1073
70
1198
130
neuronid_monitor2
41.0
1
0
Number

PLOT
0
481
1073
669
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
0
184
137
217
awakecreature?
awakecreature?
0
1
-1000

SWITCH
0
382
123
415
noxious_red
noxious_red
0
1
-1000

SWITCH
0
415
123
448
noxious_white
noxious_white
0
1
-1000

SWITCH
0
448
123
481
noxious_green
noxious_green
1
1
-1000

SWITCH
0
217
139
250
istrainingmode?
istrainingmode?
0
1
-1000

BUTTON
0
148
120
184
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
0
112
120
148
relocate insect
ask patches with [ pxcor = 102 and pycor = 30 ] [set pcolor green]\nask testcreatures [setxy 102 30]
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
0
250
148
283
insect_view_distance
insect_view_distance
1
15
4.0
1
1
NIL
HORIZONTAL

SWITCH
0
349
149
382
enable_plots?
enable_plots?
1
1
-1000

SWITCH
0
283
148
316
leave_trail_on?
leave_trail_on?
1
1
-1000

SWITCH
0
316
149
349
show_sight_line?
show_sight_line?
0
1
-1000

PLOT
727
669
1073
859
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
1073
130
1198
190
observed_efficacy_of_neuron
41.0
1
0
Number

MONITOR
1073
257
1198
302
NIL
fitness_value
17
1
11

INPUTBOX
1198
10
1269
70
weight1
-11.2
1
0
Number

INPUTBOX
1198
70
1269
130
weight2
-14.2
1
0
Number

INPUTBOX
1198
130
1269
190
weight3
-20.0
1
0
Number

INPUTBOX
1198
190
1269
250
weight4
20.0
1
0
Number

INPUTBOX
1198
250
1269
310
weight5
-2.8
1
0
Number

INPUTBOX
1198
310
1269
370
weight6
19.8
1
0
Number

INPUTBOX
1198
370
1269
430
weight7
-16.6
1
0
Number

INPUTBOX
1198
430
1269
490
weight8
8.2
1
0
Number

INPUTBOX
1198
490
1269
550
weight9
1.6
1
0
Number

INPUTBOX
1198
550
1269
610
weight10
20.0
1
0
Number

INPUTBOX
1198
610
1269
670
weight11
3.8
1
0
Number

INPUTBOX
1269
10
1340
70
weight12
-16.4
1
0
Number

INPUTBOX
1269
70
1340
130
weight13
20.0
1
0
Number

INPUTBOX
1269
130
1340
190
weight14
-20.0
1
0
Number

INPUTBOX
1269
190
1340
250
weight15
-5.6
1
0
Number

INPUTBOX
1269
250
1340
310
weight16
-16.2
1
0
Number

INPUTBOX
1269
310
1340
370
weight17
-14.0
1
0
Number

INPUTBOX
1269
370
1340
430
weight18
-5.8
1
0
Number

INPUTBOX
1269
430
1340
490
weight19
-18.8
1
0
Number

INPUTBOX
1269
490
1340
550
weight20
13.6
1
0
Number

INPUTBOX
1269
550
1340
610
weight21
20.0
1
0
Number

INPUTBOX
1269
610
1340
670
weight22
-6.6
1
0
Number

INPUTBOX
1340
10
1411
70
weight23
-13.4
1
0
Number

INPUTBOX
1340
70
1411
130
weight24
-5.6
1
0
Number

INPUTBOX
1340
130
1411
190
weight25
-8.8
1
0
Number

INPUTBOX
1340
190
1411
250
weight26
-13.0
1
0
Number

INPUTBOX
1340
250
1411
310
weight27
6.8
1
0
Number

INPUTBOX
1340
310
1411
370
weight28
12.6
1
0
Number

INPUTBOX
1340
370
1411
430
weight29
-8.0
1
0
Number

INPUTBOX
1340
430
1411
490
weight30
-14.0
1
0
Number

INPUTBOX
1340
490
1411
550
weight31
13.2
1
0
Number

INPUTBOX
1340
550
1411
610
weight32
-9.0
1
0
Number

INPUTBOX
1340
610
1411
670
weight33
-14.6
1
0
Number

INPUTBOX
1411
10
1482
70
weight34
6.0
1
0
Number

INPUTBOX
1411
70
1482
130
weight35
5.2
1
0
Number

INPUTBOX
1411
130
1482
190
weight36
-2.4
1
0
Number

INPUTBOX
1411
190
1482
250
weight37
-14.2
1
0
Number

INPUTBOX
1411
250
1482
310
weight38
5.6
1
0
Number

INPUTBOX
1411
310
1482
370
weight39
-18.2
1
0
Number

INPUTBOX
1411
370
1482
430
weight40
-4.4
1
0
Number

INPUTBOX
1411
430
1482
490
weight41
-14.2
1
0
Number

INPUTBOX
1411
490
1482
550
weight42
-19.0
1
0
Number

INPUTBOX
1411
550
1482
610
weight43
7.8
1
0
Number

INPUTBOX
1411
610
1482
670
weight44
4.6
1
0
Number

INPUTBOX
1482
10
1553
70
weight45
19.6
1
0
Number

INPUTBOX
1482
70
1553
130
weight46
19.2
1
0
Number

INPUTBOX
1482
130
1553
190
weight47
-6.8
1
0
Number

INPUTBOX
1482
190
1553
250
weight48
-20.0
1
0
Number

INPUTBOX
1482
250
1553
310
weight49
20.0
1
0
Number

INPUTBOX
1482
310
1553
370
weight50
-0.8
1
0
Number

INPUTBOX
1482
370
1553
430
weight51
-3.6
1
0
Number

INPUTBOX
1482
430
1553
490
weight52
-16.2
1
0
Number

INPUTBOX
1482
490
1553
550
weight53
15.2
1
0
Number

INPUTBOX
1482
550
1553
610
weight54
-19.2
1
0
Number

INPUTBOX
1482
610
1553
670
weight55
-20.0
1
0
Number

INPUTBOX
1553
10
1624
70
weight56
-5.6
1
0
Number

INPUTBOX
1553
70
1624
130
weight57
-17.0
1
0
Number

INPUTBOX
1553
130
1624
190
weight58
13.0
1
0
Number

INPUTBOX
1553
190
1624
250
weight59
-20.0
1
0
Number

INPUTBOX
1553
250
1624
310
weight60
-2.8
1
0
Number

INPUTBOX
1553
310
1624
370
weight61
16.2
1
0
Number

INPUTBOX
1553
370
1624
430
weight62
-18.6
1
0
Number

INPUTBOX
1553
430
1624
490
weight63
-16.2
1
0
Number

INPUTBOX
1553
490
1624
550
weight64
-20.0
1
0
Number

INPUTBOX
1553
550
1624
610
weight65
-10.8
1
0
Number

INPUTBOX
1553
610
1624
670
weight66
0.4
1
0
Number

INPUTBOX
1624
10
1695
70
weight67
-15.2
1
0
Number

INPUTBOX
1624
70
1695
130
weight68
18.4
1
0
Number

INPUTBOX
1624
130
1695
190
weight69
20.0
1
0
Number

INPUTBOX
1624
190
1695
250
weight70
-10.2
1
0
Number

INPUTBOX
1624
250
1695
310
weight71
9.2
1
0
Number

INPUTBOX
1624
310
1695
370
weight72
-11.4
1
0
Number

INPUTBOX
1624
370
1695
430
weight73
20.0
1
0
Number

INPUTBOX
1624
430
1695
490
weight74
-18.2
1
0
Number

INPUTBOX
1624
490
1695
550
weight75
4.6
1
0
Number

INPUTBOX
1624
550
1695
610
weight76
-5.2
1
0
Number

INPUTBOX
1624
610
1695
670
weight77
13.6
1
0
Number

INPUTBOX
1695
10
1766
70
weight78
-13.4
1
0
Number

INPUTBOX
1766
10
1828
70
delay1
3.0
1
0
Number

INPUTBOX
1766
70
1828
130
delay2
3.0
1
0
Number

INPUTBOX
1766
130
1828
190
delay3
6.0
1
0
Number

INPUTBOX
1766
190
1828
250
delay4
7.0
1
0
Number

INPUTBOX
1766
250
1828
310
delay5
2.0
1
0
Number

INPUTBOX
1766
310
1828
370
delay6
3.0
1
0
Number

INPUTBOX
1766
370
1828
430
delay7
4.0
1
0
Number

INPUTBOX
1766
430
1828
490
delay8
5.0
1
0
Number

INPUTBOX
1766
490
1828
550
delay9
7.0
1
0
Number

INPUTBOX
1766
550
1828
610
delay10
7.0
1
0
Number

INPUTBOX
1766
610
1828
670
delay11
1.0
1
0
Number

INPUTBOX
1828
10
1890
70
delay12
3.0
1
0
Number

INPUTBOX
1828
70
1890
130
delay13
3.0
1
0
Number

INPUTBOX
1828
130
1890
190
delay14
3.0
1
0
Number

INPUTBOX
1828
190
1890
250
delay15
2.0
1
0
Number

INPUTBOX
1828
250
1890
310
delay16
4.0
1
0
Number

INPUTBOX
1828
310
1890
370
delay17
3.0
1
0
Number

INPUTBOX
1828
370
1890
430
delay18
1.0
1
0
Number

INPUTBOX
1828
430
1890
490
delay19
6.0
1
0
Number

INPUTBOX
1828
490
1890
550
delay20
5.0
1
0
Number

INPUTBOX
1828
550
1890
610
delay21
6.0
1
0
Number

INPUTBOX
1828
610
1890
670
delay22
6.0
1
0
Number

INPUTBOX
1890
10
1952
70
delay23
7.0
1
0
Number

INPUTBOX
1890
70
1952
130
delay24
3.0
1
0
Number

INPUTBOX
1890
130
1952
190
delay25
5.0
1
0
Number

INPUTBOX
1890
190
1952
250
delay26
1.0
1
0
Number

INPUTBOX
1890
250
1952
310
delay27
6.0
1
0
Number

INPUTBOX
1890
310
1952
370
delay28
7.0
1
0
Number

INPUTBOX
1890
370
1952
430
delay29
4.0
1
0
Number

INPUTBOX
1890
430
1952
490
delay30
1.0
1
0
Number

INPUTBOX
1890
490
1952
550
delay31
2.0
1
0
Number

INPUTBOX
1890
550
1952
610
delay32
3.0
1
0
Number

INPUTBOX
1890
610
1952
670
delay33
2.0
1
0
Number

INPUTBOX
1952
10
2014
70
delay34
5.0
1
0
Number

INPUTBOX
1952
70
2014
130
delay35
3.0
1
0
Number

INPUTBOX
1952
130
2014
190
delay36
7.0
1
0
Number

INPUTBOX
1952
190
2014
250
delay37
4.0
1
0
Number

INPUTBOX
1952
250
2014
310
delay38
2.0
1
0
Number

INPUTBOX
1952
310
2014
370
delay39
7.0
1
0
Number

INPUTBOX
1952
370
2014
430
delay40
4.0
1
0
Number

INPUTBOX
1952
430
2014
490
delay41
6.0
1
0
Number

INPUTBOX
1952
490
2014
550
delay42
1.0
1
0
Number

INPUTBOX
1952
550
2014
610
delay43
4.0
1
0
Number

INPUTBOX
1952
610
2014
670
delay44
6.0
1
0
Number

INPUTBOX
2014
10
2076
70
delay45
5.0
1
0
Number

INPUTBOX
2014
70
2076
130
delay46
1.0
1
0
Number

INPUTBOX
2014
130
2076
190
delay47
4.0
1
0
Number

INPUTBOX
2014
190
2076
250
delay48
3.0
1
0
Number

INPUTBOX
2014
250
2076
310
delay49
2.0
1
0
Number

INPUTBOX
2014
310
2076
370
delay50
4.0
1
0
Number

INPUTBOX
2014
370
2076
430
delay51
3.0
1
0
Number

INPUTBOX
2014
430
2076
490
delay52
1.0
1
0
Number

INPUTBOX
2014
490
2076
550
delay53
2.0
1
0
Number

INPUTBOX
2014
550
2076
610
delay54
1.0
1
0
Number

INPUTBOX
2014
610
2076
670
delay55
6.0
1
0
Number

INPUTBOX
2076
10
2138
70
delay56
6.0
1
0
Number

INPUTBOX
2076
70
2138
130
delay57
7.0
1
0
Number

INPUTBOX
2076
130
2138
190
delay58
7.0
1
0
Number

INPUTBOX
2076
190
2138
250
delay59
4.0
1
0
Number

INPUTBOX
2076
250
2138
310
delay60
1.0
1
0
Number

INPUTBOX
2076
310
2138
370
delay61
2.0
1
0
Number

INPUTBOX
2076
370
2138
430
delay62
3.0
1
0
Number

INPUTBOX
2076
430
2138
490
delay63
6.0
1
0
Number

INPUTBOX
2076
490
2138
550
delay64
3.0
1
0
Number

INPUTBOX
2076
550
2138
610
delay65
4.0
1
0
Number

INPUTBOX
2076
610
2138
670
delay66
1.0
1
0
Number

INPUTBOX
2138
10
2200
70
delay67
1.0
1
0
Number

INPUTBOX
2138
70
2200
130
delay68
3.0
1
0
Number

INPUTBOX
2138
130
2200
190
delay69
6.0
1
0
Number

INPUTBOX
2138
190
2200
250
delay70
6.0
1
0
Number

INPUTBOX
2138
250
2200
310
delay71
4.0
1
0
Number

INPUTBOX
2138
310
2200
370
delay72
1.0
1
0
Number

INPUTBOX
2138
370
2200
430
delay73
5.0
1
0
Number

INPUTBOX
2138
430
2200
490
delay74
2.0
1
0
Number

INPUTBOX
2138
490
2200
550
delay75
3.0
1
0
Number

INPUTBOX
2138
550
2200
610
delay76
1.0
1
0
Number

INPUTBOX
2138
610
2200
670
delay77
4.0
1
0
Number

INPUTBOX
2200
10
2262
70
delay78
2.0
1
0
Number

MONITOR
1073
318
1198
363
NIL
collisions
17
1
11

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
  <experiment name="experiment" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>count turtles</metric>
    <enumeratedValueSet variable="neuronid_monitor1">
      <value value="22"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="noxious_green">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="insect_view_distance">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="enable_plots?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show_sight_line?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="neuronid_monitor2">
      <value value="21"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="noxious_white">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="noxious_red">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="istrainingmode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="leave_trail_on?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="awakecreature?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="observed_efficacy_of_neuron">
      <value value="21"/>
    </enumeratedValueSet>
  </experiment>
  <experiment name="experiment" repetitions="1" runMetricsEveryStep="true">
    <setup>setup</setup>
    <go>go</go>
    <metric>count turtles</metric>
    <enumeratedValueSet variable="neuronid_monitor1">
      <value value="22"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="noxious_green">
      <value value="false"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="insect_view_distance">
      <value value="4"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="enable_plots?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="show_sight_line?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="neuronid_monitor2">
      <value value="21"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="noxious_white">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="noxious_red">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="istrainingmode?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="leave_trail_on?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="awakecreature?">
      <value value="true"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="observed_efficacy_of_neuron">
      <value value="21"/>
    </enumeratedValueSet>
    <enumeratedValueSet variable="fitness_value">
      <value value="50"/>
    </enumeratedValueSet>
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
