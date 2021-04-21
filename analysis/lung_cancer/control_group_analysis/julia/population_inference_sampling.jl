# Infer population model parameters from control group

using Pumas, MCMCChains, StatsPlots, CSV, Random
plotly()

# Import data
path = joinpath(
    @__DIR__,
    "../../..", "python/lung_cancer/data",
    "lxf_control_growth.csv")
data = DataFrame(CSV.File(path))

# Filter Body weight information
data = filter(x -> !(occursin.("Body weight", x.Biomarker)), data)
pumas_data = read_pumas(data,
    id           = :ID,
    time         = :Time,
    observations = [:Measurement],
    event_data = false)

# Visualise tumour volume
@df data scatter(
    :Time,
    :Measurement,
    group = :ID)
yaxis!("Tumour volume in cm^3")
xaxis!("Time in day")

# Build model
tumour_growth_model = @model begin
    @param   begin
        mean_initial_volume ~ Constrained(
            Normal(0.2, 1), lower=0.001, upper=Inf, init=0.2)
        mean_critical_volume ~ Constrained(
            Normal(0.2, 1), lower=0.001, upper=Inf, init=0.2)
        mean_growth_rate ~ Constrained(
            Normal(0.2, 1), lower=0.001, upper=Inf, init=0.2)
        std ~ Constrained(
            MvNormal([1 , 1, 1]),
            lower=[0, 0, 0],
            upper=[Inf, Inf, Inf],
            init=[0.09,0.09, 0.09])
        σ_base ~ Constrained(
            Normal(0.2, 1), lower=0.001, upper=Inf, init=0.2)
        σ_rel ~ Constrained(
            Normal(0.2, 1), lower=0.001, upper=Inf, init=0.2)
        # mean_initial_volume ∈ RealDomain(lower=0, init = 0.2)
        # mean_critical_volume ∈ RealDomain(lower=0, init = 1)
        # mean_growth_rate ∈ RealDomain(lower = 0, init= 0.1)
        # std ∈ PDiagDomain(init=[0.09,0.09, 0.09])
        # σ_base ∈ RealDomain(lower=0, init=0.04)
        # σ_rel ∈ RealDomain(lower=0, init=0.04)
    end

    @random begin
        η ~ MvNormal(std)
    end

    @pre begin
        initial_volume = mean_initial_volume * exp(η[1])
        critical_volume = mean_critical_volume * exp(η[2])
        growth_rate = mean_growth_rate * exp(η[3])
    end

    @init begin
        volume = initial_volume
    end

    @dynamics begin
       volume' = (growth_rate * critical_volume * volume) / (volume + critical_volume)
    end

  @derived begin
      σ_tot := σ_base .+ volume * σ_rel
      Measurement ~ @. Normal(volume, σ_tot)
    end
end

# Infer parameters
param = init_param(tumour_growth_model)
result = fit(tumour_growth_model, pumas_data, param, Pumas.BayesMCMC();
  nsamples=2000, nadapts=1000)

# Visualise population parameters
chains = Pumas.Chains(result)
plot(chains[1000:end])
