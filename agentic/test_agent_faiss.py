from agentic.agent import run_medsimpli_agent


QUESTIONS = [
    "O que são hepatites virais?",
    "O que é hanseníase?",
    "O que é dengue?",
    "O que é Zika?",
    "Qual remédio devo tomar para dengue?",
    "Quem ganhou o jogo do Brasil?"
]


def main():
    for question in QUESTIONS:
        print("=" * 80)
        print(f"Pergunta: {question}")

        result = run_medsimpli_agent(
            user_input=question,
            use_mock=False
        )

        print(f"Intent: {result.get('intent')}")
        print(f"Status: {result.get('status')}")

        print("\nResposta:")
        print(result.get("answer"))

        print("\nValidação:")
        print(result.get("validation"))

        print("\nFontes:")
        for source in result.get("sources", []):
            print(
                f"- {source.get('source')} | "
                f"p. {source.get('page')} | "
                f"type={source.get('type')}"
            )

        print()


if __name__ == "__main__":
    main()