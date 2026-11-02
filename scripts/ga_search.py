"""Small categorical genetic algorithm; evaluator receives no test information."""
import json
import random


def key(genes):
    return json.dumps(genes, sort_keys=True, separators=(',', ':'))


def rank(record):
    score = record['fitness']
    return (-(score if score is not None else float('-inf')), record['trial_id'])


def search(space, settings, baseline_genes, evaluate, checkpoint):
    rng = random.Random(settings['search_seed'])
    names = list(space)
    size, elites = settings['population_size'], settings['elites']
    budget = size + settings['offspring_generations'] * (size - elites)
    if budget != settings['unique_trial_budget'] or not 0 < elites < size:
        raise ValueError('Invalid GA budget or elite count')
    if not 1 <= settings['tournament_size'] <= size:
        raise ValueError('Invalid tournament size')
    if any(len(values) < 2 for values in space.values()):
        raise ValueError('Each gene must have at least two alleles')
    seen, trials, generations = set(), [], []
    rejected = 0

    def random_genes():
        return {name: rng.choice(space[name]) for name in names}

    def tournament(population):
        return min(rng.sample(population, settings['tournament_size']), key=rank)

    for generation in range(settings['offspring_generations'] + 1):
        population = [] if generation == 0 else sorted(population, key=rank)[:elites]
        parents = [] if generation == 0 else previous_population
        attempts = 0
        while len(population) < size:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError('Search space exhausted or unable to propose novel offspring')
            parent_ids, mutated, crossed = [], [], False
            if generation == 0:
                genes = dict(baseline_genes) if not trials and settings['include_fixed_baseline_in_initial_population'] else random_genes()
            else:
                a, b = tournament(parents), tournament(parents)
                parent_ids = [a['trial_id'], b['trial_id']]
                crossed = rng.random() < settings['crossover_probability']
                genes = {name: (rng.choice([a['genes'][name], b['genes'][name]]) if crossed else a['genes'][name]) for name in names}
                for name in names:
                    if rng.random() < settings['mutation_probability_per_gene']:
                        genes[name] = rng.choice([v for v in space[name] if v != genes[name]])
                        mutated.append(name)
            signature = key(genes)
            if signature in seen:
                rejected += 1
                continue
            seen.add(signature)
            trial_id = len(trials)
            result = evaluate(genes, trial_id, generation)
            record = {**result, 'trial_id': trial_id, 'generation': generation, 'genes': genes,
                      'parents': parent_ids, 'crossover_applied': crossed, 'mutated_genes': mutated}
            trials.append(record)
            population.append(record)
            checkpoint({'trials': trials, 'generations': generations, 'duplicate_proposals_rejected': rejected})
        previous_population = list(population)
        generations.append({'generation': generation, 'population_trial_ids': [r['trial_id'] for r in population],
                            'best_trial_id': min(population, key=rank)['trial_id']})
        checkpoint({'trials': trials, 'generations': generations, 'duplicate_proposals_rejected': rejected})
    return sorted(trials, key=rank)
