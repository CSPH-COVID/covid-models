# Colorado COVID-19 Model

This repo contains a set of tools for running and processing the State of Colorado COVID-19 model, designed by the Colorado COVID-19 Modeling Group. 
Previous versions of these models can be found in [another repo](https://github.com/agb85/covid-19).
Periodic modeling reports can be found on the Colorado School of Public Health website [here](https://coloradosph.cuanschutz.edu/resources/covid-19/modeling-results).

# Model Overview

At its core, the model is [compartmental](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology), where individuals move between different _compartments_, governed by a system of ordinary differential equations.
The term _individual_ is perhaps misleading, as the model doesn't track individuals, instead it tracks the number of people in a given compartment (state).
Moreover the counts and flows of people are often not integers, which is an artifact of the choice to model using differential equations.

### Compartment Structure

The compartments themselves are defined by a set of __attributes__, which capture different aspects of a population's state.
Each compartment is a unique combination of the attributes in the model, and the set of all compartments is the cartesian product of the possible values for each attribute.
Here is a list of the current attributes in the model, followed by the possible values for each attribute.
* __Disease Status__, can be S (Susceptible), E (Exposed), I (Infectous Symptomatic), A (Infectous Asymptomatic), Ih (Infectious Symptomatic and Hospitalized), D (Deceased). Note: we do not have a "recovered" level, individuals become susceptible immediately after infection has run its course, though they will have some immunity afterwards.
* __Age__, bracketed into four ranges: 0-19, 20-39, 40-64, and 65+
* __Vaccination status__: None, Shot1 (one J&J or MRNA shot), Shot2 (two MRNA shots), Shot3 (booster of any type)
* __Variant__, either of current infection if SEIR status is E, I, A, or Ih, or prior infected variant if SEIR status is S or D; Variants currently in the model are None (no current or prior infection), Wildtype, Alpha, Delta, Omicron (BA.1), Omicron BA.2, Omicron BA.2.12.1, and Omicron BA.4/5 (representing BA.4 and BA.5 combined) 
* __Immunity level__, representing an individual's protection against infection and severe infection; can be None, Weak, or Strong (more detail below)
* __Region__, useful when modeling separate, interacting regions; this is always "CO" when running the model on the state of Colorado

Compartments can be referenced with a tuple specifying the value for each attribute, for example: _(S, 20-39, Shot3, Omicron, Strong, CO)_ is the compartment for susceptible individuals aged 20-39 who have received a booster shot, have a prior Omicron BA.1 infection, have strong immunity, and reside in Colorado.
The count for this compartment indicates how much of the population possesses these attribute values.

TODO: Add some other examples of compartments.

A _state_ refers to a full specification of the count in each compartment of the model.

### Model Dynamics

Starting from an initial state, a (deterministic) timeseries for each compartment can be determined by numerically solving the system of differential equations.
We can also conceptualize the set of compartment timeseries as a continuous flow of people between different departments.
For example, to represent susceptible individuals becoming infected, we have a flow from each "S" compartment to its corresponding "E" compartment, where the rate of flow is determined by the transmission characteristics of the relevant variant, as well as the number of individuals in the appropriate "I" and "A" compartments. 

In lieu of the full set of differential equations governing (coming soon), we can describe the model dynamics pictorally using a set of flow diagrams.
TODO: Add in flow diagrams and differential equations.

# Repository Information

This repository contains the core model code, along with many relevant model parameters.
It also contains code to read/write data to a database, fit and run the model, and run various analyses using the model.

### Organization

TODO: explain file structure, etc.

### Model Parameters

Much of the model's behavior is governed by parameters, for example, the total population, the infectiousness of a particular variant, and the immunity conferred by vaccination.
These parameters are allowed to change in time, and all fall into one of the following categories:
* Global: these parameters apply to all compartments of the model. There aren't many global parameters, and in practice global parameters are often modified for a subset of compartments, effectively becoming not global anymore. An example of a global paramter is the rate of spread of the wildtype variant, aka. "betta"[1].
* Associated with a compartment: These parameters are associated to a particular compartment. For example, the hospitalization rate from a particular variant may differ for those aged 20-39 vs. those aged 65+.
* Associated with a flow: These parameters are associated to a particular flow between two compartments. For example, the flow from a particular susceptible compartment to a particular exposed compartment may depend the previous variant of infection specified in the susceptible compartment, and the next variant specified in the exposed compartment (this is to capture the differing immune escape that each variant has to a particular prior variant infection). 

Parameters are mostly specified in a `json` file, which is just a __list of specifications__, each of which is a "json object" (which becomes a dictionary once loaded into Python; in other words, it's a set of key-value pairs).
Each specification either __assigns a value to a parameter__ or __modifies an exising value of a parameter__, for a certain subset of compartments or compartment pairs.
These specifications have standardized formats, and must have certain keys to be valid, depending on whether the parameter is global, associated with a compartment, or associated with compartment pairs.

#### Examples of Compartment-Associated Parameters:

Here is an example of a parameter associated with a subset of compartments:

```python
{
    "param": "region_age_pop",	
    "attrs": {"region":  "co", "age": "0-19"},	
    "vals": {"2020-01-01": 1411161},	
    "desc": "the population of individuals aged 0-19 in CO"
}
```
Let's talk through each key in this specification
* `"param"` gives the name of the parameter we are specifying, in this case, that's `"region_age_pop"`
* `"attrs"` specifies one or more attributes that are used as a _filter_ to determine which compartments this specification applies to. In this case, any compartment where the "region" attribute is "CO" and the "age" attribute is "0-19" will be affected by this specification.
* `"vals"` gives the values of the parameter at specified dates. Parameters are assumed constant in time until changed by a new date entry. In this case, the population is set to 1,411,161 starting January 1st 2020, and is not changed at any later date, so it is fixed for the whole modeling period.
* `"desc"` gives a description of the parameter specification, to help explain what it is doing and why in human language.

Here is a related specification, which applies to the same parameter (`"region_age_pop"`), but affects a different subset of compartments (individuals in Colorado who are aged 20-39):
```python
{
    "param": "region_age_pop",	
    "attrs": {"region":  "co", "age": "20-39"},	
    "vals": {"2020-01-01": 1697671},
    "desc": "the population of individuals aged 20-39 in CO"
}
 ```
You can see that this value of the parameter is different, reflecting the different population of this age group in this region.

Specifications may apply __multipliers__ to parameter values rather than set the value explicitly. 
This is useful, for example, when we want to define the parameters for a new COVID-19 variant relative to some prior variant.
Here is an example:
```python
{
    "param": "betta", 
    "attrs": {"variant": "alpha"}, 
    "mults": {"2020-01-01": 1.5},
    "desc": "Alpha is more infectious than wildtype."
}
```
This specification applies to the `"betta"` parameter, which controls the rate of disease transmission, and the `"attrs"` key makes this specification apply only to compartments where the variant attribute is "alpha".
Instead of a `"vals"` key, this specification has a `"mults"` key, which says to take the current value of this parameter and multiply it by the given parameter on the given dates.
As with `"vals"`, each multiplier is assumed constant in time until changed.
In this case, the multiplier is 1.5 from Jan 1st 2020 onward, so it is fixed for the whole modeling period.
All together, this specification alters the `"betta"` parameter for the alpha variant to be 1.5 times the default betta (which is taken to be the betta for the wildtype variant).

The examples so far have set or adjusted parameters for one date, but here is an example where a parameter changes in time:
```python
{
    "param": "hosp",
    "attrs": {"region": "co", "age": "0-19"},
    "vals": {"2020-01-01": 0.0286, "2020-09-30": 0.0223},
    "desc": "hosp rate decreased in September for 0-19 year olds in CO"
}
```

Here, the `"hosp"` parameter, which specifies the rate of hospitalization for infected individuals is being set for individuals aged 0-19 in Colorado.
The value is set to `0.0286` on Jan 1st 2020, and is assumed constant until it changes to `0.0223` on September 30 2020, and is assumed constant after that.

Finally, here is an example showing a more sophistocated way of filtering compartments using the `"attrs"` key:
```python
{
    "param": "alpha",
    "attrs": {"variant": ["omicron", "ba2", "ba2121", "ba45"]},
    "mults": {"2020-01-01": 0.6},
    "desc": "Omicron has shorter incubation period than wildtype."
}
```
Notice that a list is provided for the `"variant"` attribute.
This means that this specification applies to compartments matching _any_ of the prescribed variants.
Using this notation, we can condense four separate specifications (one for each variant) into a single specification to save space and help avoid consistency errors when updating parameters.

#### Example of Flow-Associated Parameters:

There aren't that many parameters assigned to pairs of compartments, but this capability is particularly useful to capture immune escape. 
Here is an example of a specification for a parameter associated with pairs of compartments:

```python
{
    "param": "immune_escape",
    "from_attrs": {"immun": "strong", "variant": ["omicron", "ba2"]},
    "to_attrs": {"variant": ["ba2121"]},
    "vals": {"2020-01-01": 0.1},
    "desc": "the BA.2.12.1 variant has some immune escape against prior Omicron BA.1 and BA.2 infection"
}
```

Instead of a `"attrs"` key, this specification has a `"from_attrs"` key and a `"to_attrs"` key, which collectively define to which flows this specification applies.
The "from" and "to" nomenclature reflects the fact that every flow if _from_ some compartment _to_ some other compartment, so this specification only aplies to one direction.
The `"from_attrs"` item works the same way as the `"attrs"` item in a compartment-attached specification: only flows with a _from compartment_ matching these attributes are considered by this specification
In addition, the _to compartment_ of flows must match the `"to_attrs"` attributes as well, but the `"to_attrs"` item works slightly differently.
It specifies the _change_ that occurs between the _from compartment_ and _to compartment_.
In the above example, `"from_attrs"` implies that any flow where the _from compartment_ has strong immunity, and either Omicron BA.1 or BA.2 variants, should be considered.
The `"to_attrs"` implies that only flows where everything between the _from compartment_ and _to compartment_ is the same except the variant in the _to compartment_ is Omicron BA.2.12.1, are affected by this specification.
That means the flow (S, 0-19, Shot1, omicron, weak, CO) -> (E, 0-19, Shot1, ba2121, weak, CO) is affected by this specification, but the following flows are __not__ affected by this specification:
* (S, 0-19, Shot1, wildtype, weak, CO) -> (E, 0-19, Shot1, ba2121, weak, CO) (does not meet the `"from_attrs"` requirements)
* (S, 0-19, Shot1, omicron, weak, CO) -> (E, 0-19, Shot2, omicron, weak, CO) (the vaccine status changes, but `"to_attrs"` only allows the variant to change to "ba2121")
* (S, 0-19, Shot1, omicron, weak, CO) -> (E, 0-19, Shot2, ba2121, weak, CO) (here the variant has changed correctly, but again the vaccine status has changed, which is not allowed)
The rest of the keys in this specification are the same: `"param"` specifies which parameter is affected, either `"vals"` or `"mults"` will specify how that parameter is to be set or adjusted through time, and `"desc"` gives a description of what the specification is trying to accomplish.

#### Global Parameters:

For compartment-associated parameters, global parameters can be set by either specifying `null` for the `"attrs"` key or an empty json object (i.e. python dictionary), like so `{}`[2].
Here is an example:
```python
{
    "param": "imm_decay_days",
    "attrs": null,
    "vals": {"2020-01-01": 360},
    "desc": "The average time for immunity to decay is about a year"
}
```

For flow-associated parameters, we can set the `"from_attrs"` and `"to_attrs"` keys to `null` or `{}` like so:

```python
{
    "param": "immune_escape", 
    "from_attrs": null, 
    "to_attrs": null,
    "vals": {"2020-01-01": 0},  
    "desc":  ""
}
```
 

#### How Specifications are Processed

The order of specifications in the `json` file matters, because specifications are applied in order.
For example, a parameter may be set globally for all compartments, then later a multiplier might be applied to that value for a subset of compartments.
Reverseing the order would cause multiple issues: (1) a multiplier can't be applied to a parameter which hasn't been given some value already, and (2) setting the value of a parameter will override any values or multipliers that have been applied to that parameter previously. 

#### Footnotes

[1] In our model we use [Sympy](https://www.sympy.org/en/index.html) to parse parameters, and 'beta' has built in meaning. So we use 'betta' instead to refer to the infectiousness parameter, and everything has worked swimmingly ;).

[2] Stricly speaking, `null` and `{}` put data in slightly different places, but the effect is the same. `{}` actually applies the specification to every compartment individually, whereas `null` stores the value once, outside the compartments, which is checked whenever a parameter is needed for a particular compartment but hasn't been set yet. This should not concern the casual user of the model.

### The Model Class

TODO: talk about the model class. instantiation, prep, solving, serialization, etc.

# Working with the model

TODO:

### Data Dependencies

TODO: describe database

### Environment setup

TODO: packages and environment variables

### Common tasks

TODO: fit the model, run report, etc. Stuff in `runnable_functions.py`
