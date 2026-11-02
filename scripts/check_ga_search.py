"""Check real GA invariants with a cheap objective, then model-space gradients."""
import json
from pathlib import Path
import unittest
from ga_search import search, key, rank

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / 'configs/ga_lstm_xlstm_comparison.json').read_text())
BASE = json.loads((ROOT / 'configs/lstm_xlstm_comparison.json').read_text())


class GeneticAlgorithmChecks(unittest.TestCase):
    def run_search(self, reverse=False, fail=False):
        states = []
        def evaluator(genes, trial, generation):
            fitness = sum(SPEC['search_space'][name].index(value) for name, value in genes.items())
            return {'fitness': None if fail and trial == 0 else (-fitness if reverse else fitness)}
        result = search(SPEC['search_space'], SPEC['ga'], {k: BASE[k] for k in SPEC['search_space']},
                        evaluator, lambda state: states.append(json.loads(json.dumps(state))))
        return result, states[-1]

    def test_budget_uniqueness_elitism_and_lineage(self):
        result, state = self.run_search()
        self.assertEqual(len(result), 18)
        self.assertEqual(len({key(r['genes']) for r in result}), 18)
        self.assertEqual(len(state['generations']), 4)
        by_id = {r['trial_id']: r for r in result}
        for gen in state['generations']:
            ids = gen['population_trial_ids']
            self.assertEqual(len(set(ids)), 6)
            if gen['generation']:
                previous = state['generations'][gen['generation'] - 1]['population_trial_ids']
                elites = sorted((by_id[i] for i in previous), key=rank)[:2]
                self.assertTrue({r['trial_id'] for r in elites}.issubset(ids))
        for row in result:
            for name, value in row['genes'].items():
                self.assertIn(value, SPEC['search_space'][name])
            if row['generation']:
                self.assertEqual(len(row['parents']), 2)
                self.assertTrue(all(parent < row['trial_id'] for parent in row['parents']))
        self.assertTrue(any(row['mutated_genes'] for row in result))

    def test_replay_and_common_initial_population(self):
        first, a = self.run_search()
        second, b = self.run_search()
        self.assertEqual(first, second)
        self.assertEqual(a, b)
        _, reverse = self.run_search(reverse=True)
        self.assertEqual([r['genes'] for r in a['trials'][:6]], [r['genes'] for r in reverse['trials'][:6]])
        self.assertNotEqual([r['genes'] for r in a['trials'][6:]], [r['genes'] for r in reverse['trials'][6:]])

    def test_failure_consumes_budget_and_ranks_last(self):
        result, _ = self.run_search(fail=True)
        self.assertEqual(len(result), 18)
        self.assertEqual(result[-1]['trial_id'], 0)
        self.assertIsNone(result[-1]['fitness'])


def check_models():
    from compare_lstm_xlstm import Classifier, torch
    torch.set_num_threads(4)
    for kind in SPEC['models']:
        for dim in SPEC['search_space']['embedding_dim']:
            for depth in SPEC['search_space']['num_layers']:
                model = Classifier(kind, 100, {**BASE, 'embedding_dim': dim, 'num_layers': depth})
                ids = torch.tensor([[3, 4, 2, 0], [7, 8, 9, 2]])
                loss = torch.nn.functional.cross_entropy(model(ids), torch.tensor([1, 6]))
                loss.backward()
                assert torch.isfinite(loss)
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print('All 8 architecture/width/depth combinations have finite forward/backward gradients.')


if __name__ == '__main__':
    outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GeneticAlgorithmChecks))
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    check_models()
