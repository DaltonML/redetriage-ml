
import unittest
from src.revisao import exige_revisao


class TesteRevisao(unittest.TestCase):

    def test_previsao_confiavel(self):
        """Uma categoria claramente dominante não exige revisão."""
        pontuacoes = [0.65, 0.20, 0.10, 0.05]

        self.assertFalse(exige_revisao(pontuacoes))

    def test_pontuacao_baixa(self):
        """Pontuação máxima abaixo de 30% exige revisão."""
        pontuacoes = [0.28, 0.26, 0.24, 0.22]

        self.assertTrue(exige_revisao(pontuacoes))

    def test_categorias_muito_proximas(self):
        """Diferença pequena entre categorias exige revisão."""
        pontuacoes = [0.42, 0.40, 0.10, 0.08]

        self.assertTrue(exige_revisao(pontuacoes))

    def test_numero_insuficiente_de_categorias(self):
        """A função deve rejeitar apenas uma categoria."""
        with self.assertRaises(ValueError):
            exige_revisao([1.0])


if __name__ == "__main__":
    unittest.main()
