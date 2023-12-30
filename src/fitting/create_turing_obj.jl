###########################
### FITTING A DATAFRAME ###
###########################
"""""
    fit_model(agent::Agent, inputs::Array, actions::Vector, priors::Dict, kwargs...)

Use Turing to fit the parameters of an agent to a set of inputs and corresponding actions.

# Arguments
 - 'agent::Agent': an ActionModels agent object created with either premade_agent or init_agent.
 - 'priors::Dict': dictionary containing priors (as Distribution objects) for fitted parameters. Keys are parameter names, values are priors.
 - 'inputs:Array': array of inputs. Each row is a timestep, and each column is a single input value.
 - 'actions::Array': array of actions. Each row is a timestep, and each column is a single action.
 - 'fixed_parameters::Dict = Dict()': dictionary containing parameter values for parameters that are not fitted. Keys are parameter names, values are priors. For parameters not specified here and without priors, the parameter values of the agent are used instead.
 - 'sampler = NUTS()': specify the type of Turing sampler.
 - 'n_cores = 1': set number of cores to use for parallelization. If set to 1, no parallelization is used.
 - 'n_iterations = 1000': set number of iterations per chain.
 - 'n_chains = 2': set number of amount of chains.
 - 'verbose = true': set to false to hide warnings.
 - 'show_sample_rejections = false': set whether to show warnings whenever samples are rejected.
 - 'impute_missing_actions = false': set whether the values of missing actions should also be estimated by Turing.

 # Examples
```julia
#Create a premade agent: binary Rescorla-Wagner
agent = premade_agent("premade_binary_rw_softmax")

#Set priors for the learning rate
priors = Dict("learning_rate" => Uniform(0, 1))

#Set inputs and actions
inputs = [1, 0, 1]
actions = [1, 1, 0]

#Fit the model
fit_model(agent, priors, inputs, actions, n_chains = 1, n_iterations = 10)
```
"""
function create_turing_obj(
    agent::Agent,
    priors::Dict,
    data::DataFrame;
    independent_group_cols::Vector = [],
    multilevel_group_cols::Vector = [],
    input_cols::Vector = [:input],
    action_cols::Vector = [:action],
    fixed_parameters::Dict = Dict(),
    sampler = NUTS(),
    n_cores::Integer = 1,
    n_iterations::Integer = 1000,
    n_chains::Integer = 2,
    verbose::Bool = true,
    show_sample_rejections::Bool = false,
    impute_missing_actions::Bool = false,
    sampler_kwargs...,
)
    ### SETUP ###

    #Convert column names to symbols
    independent_group_cols = Symbol.(independent_group_cols)
    multilevel_group_cols = Symbol.(multilevel_group_cols)
    input_cols = Symbol.(input_cols)
    action_cols = Symbol.(action_cols)

    ## Store old parameters for resetting the agent later ##
    old_parameters = get_parameters(agent)

    ## Set fixed parameters to agent ##
    set_parameters!(agent, fixed_parameters)

    ## Run checks ##
    prefit_checks(
        agent = agent,
        data = data,
        priors = priors,
        independent_group_cols = independent_group_cols,
        multilevel_group_cols = multilevel_group_cols,
        input_cols = input_cols,
        action_cols = action_cols,
        fixed_parameters = fixed_parameters,
        old_parameters = old_parameters,
        n_cores = n_cores,
        verbose = verbose,
    )

    ## Set logger ##
    #If sample rejection warnings are to be shown
    if show_sample_rejections
        #Use a standard logger
        sampling_logger = Logging.SimpleLogger()
    else
        #Use a logger which ignores messages below error level
        sampling_logger = Logging.SimpleLogger(Logging.Error)
    end

    ## Store whether there are multiple inputs and actions ##
    multiple_inputs = length(input_cols) > 1
    multiple_actions = length(action_cols) > 1
    multilevel = length(multilevel_group_cols) > 0

    ## Structure multilevel parameter information ##
    general_parameters_info = extract_structured_parameter_info(;
        priors = priors,
        multilevel_group_cols = multilevel_group_cols,
    )

    ## Structure data ##
    #Group data into independent groups
    independence_grouped_dataframe = groupby(data, independent_group_cols)

    #Initialize vectors of independent datasets and their keys
    independent_groups_keys = []
    independent_groups_info = []

    #Go through data for each independent group
    for (independent_group_key, independent_group_data) in
        pairs(independence_grouped_dataframe)

        ## Get the key for the independent group ##
        #If there is only independent group distinction
        if length(independent_group_cols) == 1

            #Get out that group level as key
            independent_group_key = independent_group_data[1, first(independent_group_cols)]

            #If there are multiple
        else
            #Save the key for the independent group as a tuple
            independent_group_key = Tuple(independent_group_data[1, independent_group_cols])
        end

        #Add it as a key
        push!(independent_groups_keys, independent_group_key)

        #Extract and save data as dicts of multilevel grouped arrays
        push!(
            independent_groups_info,
            extract_structured_data(
                data = independent_group_data,
                multilevel_group_cols = multilevel_group_cols,
                input_cols = input_cols,
                action_cols = action_cols,
                general_parameters_info = general_parameters_info,
            ),
        )
    end

    ## Copy the fitting info for each chain that is to be sampled ##
    fit_info_all = repeat(independent_groups_info, n_chains)

    turing_model = map(
        fit_info -> 
            create_agent_model(
                agent,
                fit_info.multilevel_parameters_info,
                fit_info.agent_parameters_info,
                fit_info.inputs,
                fit_info.actions,
                fit_info.multilevel_groups,
                multiple_inputs,
                multiple_actions,
                impute_missing_actions,
            ),
        fit_info_all,
    )

    return turing_model
end

