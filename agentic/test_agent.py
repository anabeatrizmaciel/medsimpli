from agentic.agent import run_medsimpli_agent


TEST_CASES = [
    {
        "question": "O que é dengue?",
        "expected_intent": "PERGUNTA_SAUDE",
        "expected_status": "answered"
    },
    {
        "question": "O que significa CID?",
        "expected_intent": "EXPLICAR_TERMO",
        "expected_status": "answered"
    },
    {
        "question": "Como a malária é transmitida?",
        "expected_intent": "PERGUNTA_SAUDE",
        "expected_status": "answered"
    },
    {
        "question": "Simplifique: paciente apresenta suspeita de arbovirose.",
        "expected_intent": "SIMPLIFICAR_TEXTO_MEDICO",
        "expected_status": "answered"
    },
    {
        "question": "Qual remédio devo tomar para dengue?",
        "expected_intent": "PERGUNTA_SENSIVEL",
        "expected_status": "fallback"
    },
    {
        "question": "Quem ganhou o jogo do Brasil?",
        "expected_intent": "FORA_DO_DOMINIO",
        "expected_status": "fallback"
    }
]


def run_tests():
    total = len(TEST_CASES)
    passed = 0

    for index, case in enumerate(TEST_CASES, start=1):
        result = run_medsimpli_agent(case["question"], use_mock=True)

        got_intent = result.get("intent")
        got_status = result.get("status")

        intent_ok = got_intent == case["expected_intent"]
        status_ok = got_status == case["expected_status"]

        if intent_ok and status_ok:
            passed += 1

        print("=" * 80)
        print(f"Teste {index}")
        print(f"Pergunta: {case['question']}")
        print(f"Intent esperado: {case['expected_intent']} | obtido: {got_intent}")
        print(f"Status esperado: {case['expected_status']} | obtido: {got_status}")
        print(f"Resultado: {'OK' if intent_ok and status_ok else 'FALHOU'}")
        print("-" * 80)
        print(result.get("answer"))
        print()

    print("=" * 80)
    print(f"Resumo: {passed}/{total} testes passaram.")


if __name__ == "__main__":
    run_tests()