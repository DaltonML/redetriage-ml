import unittest
from src.modelo import carregar_dados, novo_modelo, prever


class TesteModelo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        textos, categorias = carregar_dados()
        cls.modelo = novo_modelo().fit(textos, categorias)

    def test_base_tem_cinco_classes(self):
        self.assertEqual(len(self.modelo.classes_), 5)

    def test_frase_sem_texto_gera_erro(self):
        with self.assertRaises(ValueError):
            prever(self.modelo, "  ")

    def test_probabilidades_somam_um(self):
        ranking = prever(self.modelo, "Não consigo obter IP por DHCP")
        self.assertAlmostEqual(sum(escore for _, escore in ranking), 1.0)

    def test_dhcp(self):
        ranking = prever(self.modelo, "Meu computador não recebe IP do servidor DHCP")
        self.assertEqual(ranking[0][0], "DHCP")

    def test_dns(self):
        ranking = prever(self.modelo, "O nome do domínio não resolve no DNS")
        self.assertEqual(ranking[0][0], "DNS")


if __name__ == "__main__":
    unittest.main()
