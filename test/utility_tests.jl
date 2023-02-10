using ActionModels
using Distributions

#Agent
agent = premade_agent("premade_binary_rw_softmax")

#Variations of get_states
get_states(agent)

get_states(agent, "transformed_value")

get_states(agent, ["transformed_value", "action"])

#Variations of get_parameters
get_parameters(agent)

get_parameters(agent, ("initial", "value"))

get_parameters(agent, [("initial", "value"), "learning_rate"])

#Variations of set_parameters
set_parameters!(agent, ("initial", "value"), 1)

set_parameters!(agent, Dict("learning_rate" => 3, "softmax_action_precision" => 0.5))

#Variations of get_history
get_history(agent, "value")

get_history(agent)

reset!(agent)
