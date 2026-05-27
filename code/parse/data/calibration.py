"""
Calibration data construction for CIT computation.

Provides per-axis prompt templates (language, discipline, scenario) with
10-20 samples each. These compact datasets are used to compute marginal CIT
values efficiently without running full evaluation benchmarks.

Following the paper's Section 3.5 Stage 1 (Diagnostic Probing), calibration
data includes declarative, interrogative, and imperative constructions for
language coverage, and domain-specific prompts for discipline/scenario axes.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class CalibrationData:
    """Structured calibration data across the three capability axes."""

    lang: Dict[str, List[str]] = field(default_factory=dict)
    disc: Dict[str, List[str]] = field(default_factory=dict)
    scen: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def total_categories(self) -> int:
        return len(self.lang) + len(self.disc) + len(self.scen)

    @property
    def total_samples(self) -> int:
        return (
            sum(len(v) for v in self.lang.values())
            + sum(len(v) for v in self.disc.values())
            + sum(len(v) for v in self.scen.values())
        )


def load_calibration(path: str) -> CalibrationData:
    """Load calibration data from a JSON file.

    Expected format:
    {
        "lang": {"zh": ["prompt1", ...], "en": [...]},
        "disc": {"math": [...], ...},
        "scen": {"fc": [...], ...}
    }
    """
    import json
    with open(path, "r") as f:
        raw = json.load(f)
    return CalibrationData(
        lang=raw.get("lang", {}),
        disc=raw.get("disc", {}),
        scen=raw.get("scen", {}),
    )


def build_default_calibration() -> CalibrationData:
    """
    Build the default compact calibration dataset used in the paper.

    Each category contains 15 samples covering varied constructions
    (declarative, interrogative, imperative for language; factual recall,
    reasoning, problem-solving for discipline; task-specific prompts for scenario).
    """
    return CalibrationData(
        lang={
            "zh": [
                "请用中文写一段关于人工智能发展的短文。",
                "今天的天气怎么样？请用中文回答。",
                "用中文解释量子计算的基本原理。",
                "鲁迅是中国现代文学的重要作家，请介绍他的代表作。",
                "用中文翻译以下英文句子：The quick brown fox jumps over the lazy dog.",
                "请用中文概括《红楼梦》的主要情节。",
                "描述一下你理想中的城市生活。",
                "什么是机器学习？请用中文给出定义。",
                "用中文写一首五言绝句。",
                "如何用中文表达'I would like to order a coffee'？",
                "请用中文解释牛顿第二定律。",
                "长江是中国最长的河流，请介绍它的流域范围。",
                "用中文写一个关于友谊的故事。",
                "请用中文列举三种可再生能源。",
                "什么是云计算？用中文解释其优缺点。",
            ],
            "en": [
                "Write a short essay about the future of artificial intelligence.",
                "What is the weather like today? Answer in English.",
                "Explain the basic principles of quantum computing in English.",
                "Describe the plot of Shakespeare's Hamlet.",
                "Translate the following Chinese sentence to English: 人工智能正在改变世界。",
                "Summarize the key events of World War II.",
                "Describe your ideal city of the future.",
                "What is machine learning? Provide a definition in English.",
                "Write a haiku about nature.",
                "How would you politely decline an invitation in English?",
                "Explain Newton's second law of motion in simple terms.",
                "The Amazon River is the world's largest by volume. Describe its ecosystem.",
                "Write a short story about perseverance.",
                "List three benefits of renewable energy in English.",
                "What is cloud computing? Explain its pros and cons.",
            ],
            "ja": [
                "人工知能の未来について短い文章を書いてください。",
                "今日の天気はどうですか？日本語で答えてください。",
                "量子コンピュータの基本原理を日本語で説明してください。",
                "村上春樹の代表作を紹介してください。",
                "この英文を日本語に翻訳してください：AI is transforming society.",
                "明治維新の主要な出来事を要約してください。",
                "理想的な未来都市を説明してください。",
                "機械学習とは何ですか？日本語で定義を述べてください。",
                "自然についての俳句を書いてください。",
                "日本語で丁寧に招待を断る方法を教えてください。",
                "日本の四季の特徴について説明してください。",
                "日本の伝統的な祭りについて紹介してください。",
                "日本語でビジネスメールの書き方を説明してください。",
                "和食の特徴と代表的な料理について説明してください。",
                "日本の教育制度について簡単に説明してください。",
            ],
            "fr": [
                "Écrivez un court essai sur l'avenir de l'intelligence artificielle.",
                "Quel temps fait-il aujourd'hui ? Répondez en français.",
                "Expliquez les principes de base de l'informatique quantique en français.",
                "Décrivez l'intrigue des Misérables de Victor Hugo.",
                "Traduisez cette phrase en français : The Earth revolves around the sun.",
                "Qu'est-ce que l'apprentissage automatique ? Donnez une définition.",
                "Décrivez votre ville idéale du futur.",
                "Comment refuser poliment une invitation en français ?",
                "Expliquez la deuxième loi de Newton en termes simples.",
                "Énumérez trois avantages des énergies renouvelables.",
                "Qu'est-ce que le cloud computing ? Expliquez ses avantages et inconvénients.",
                "Parlez de l'importance de la Révolution française.",
                "Décrivez la cuisine française et ses plats emblématiques.",
                "Expliquez le système éducatif français.",
                "Quels sont les grands défis environnementaux actuels ?",
            ],
            "de": [
                "Schreiben Sie einen kurzen Aufsatz über die Zukunft der künstlichen Intelligenz.",
                "Wie ist das Wetter heute? Antworten Sie auf Deutsch.",
                "Erklären Sie die Grundprinzipien des Quantencomputings auf Deutsch.",
                "Beschreiben Sie die Handlung von Goethes Faust.",
                "Übersetzen Sie diesen Satz ins Deutsche: Renewable energy is the future.",
                "Was ist maschinelles Lernen? Geben Sie eine Definition auf Deutsch.",
                "Beschreiben Sie Ihre ideale Stadt der Zukunft.",
                "Wie lehnt man eine Einladung auf Deutsch höflich ab?",
                "Erklären Sie Newtons zweites Gesetz in einfachen Worten.",
                "Nennen Sie drei Vorteile erneuerbarer Energien.",
                "Was ist Cloud Computing? Erklären Sie Vor- und Nachteile.",
                "Beschreiben Sie die deutsche Kultur und Traditionen.",
                "Erklären Sie das deutsche Bildungssystem.",
                "Was sind die wichtigsten Umweltherausforderungen heute?",
                "Sprechen Sie über die Bedeutung der europäischen Union.",
            ],
            "ru": [
                "Напишите короткое эссе о будущем искусственного интеллекта.",
                "Какая сегодня погода? Ответьте на русском языке.",
                "Объясните основные принципы квантовых вычислений на русском.",
                "Опишите сюжет романа 'Война и мир' Толстого.",
                "Переведите фразу на русский: Technology changes the world.",
                "Что такое машинное обучение? Дайте определение на русском.",
                "Опишите ваш идеальный город будущего.",
                "Как вежливо отказаться от приглашения на русском?",
                "Объясните второй закон Ньютона простыми словами.",
                "Назовите три преимущества возобновляемой энергии.",
                "Что такое облачные вычисления? Объясните плюсы и минусы.",
                "Расскажите о русской культуре и традициях.",
                "Объясните систему образования в России.",
                "Каковы главные экологические проблемы современности?",
                "Расскажите о значении освоения космоса.",
            ],
            "es": [
                "Escribe un breve ensayo sobre el futuro de la inteligencia artificial.",
                "¿Qué tiempo hace hoy? Responde en español.",
                "Explica los principios básicos de la computación cuántica en español.",
                "Describe la trama de Don Quijote de Cervantes.",
                "Traduce esta frase al español: Artificial intelligence is transforming society.",
                "¿Qué es el aprendizaje automático? Da una definición en español.",
                "Describe tu ciudad ideal del futuro.",
                "¿Cómo rechazar educadamente una invitación en español?",
                "Explica la segunda ley de Newton en términos sencillos.",
                "Enumera tres beneficios de las energías renovables.",
                "¿Qué es la computación en la nube? Explica sus ventajas y desventajas.",
                "Habla sobre la importancia de la cultura hispana.",
                "Describe el sistema educativo en España.",
                "¿Cuáles son los principales desafíos ambientales actuales?",
                "Habla sobre la importancia de la literatura en español.",
            ],
            "ko": [
                "인공지능의 미래에 대한 짧은 에세이를 작성하세요.",
                "오늘 날씨는 어떻습니까? 한국어로 답변하세요.",
                "양자 컴퓨팅의 기본 원리를 한국어로 설명하세요.",
                "한국의 전통 문화에 대해 설명해 주세요.",
                "이 문장을 한국어로 번역하세요: Technology changes the world.",
                "기계 학습이란 무엇입니까? 한국어로 정의를 말하세요.",
                "당신의 이상적인 미래 도시를 설명하세요.",
                "한국어로 정중하게 초대를 거절하는 방법을 알려주세요.",
                "뉴턴의 제2법칙을 간단한 말로 설명하세요.",
                "재생 가능 에너지의 세 가지 이점을 나열하세요.",
                "클라우드 컴퓨팅이란 무엇입니까? 장단점을 설명하세요.",
                "한국의 교육 제도에 대해 설명해 주세요.",
                "한식의 특징과 대표적인 요리를 소개해 주세요.",
                "현재 가장 중요한 환경 문제는 무엇입니까?",
                "한국의 계절별 특징에 대해 설명해 주세요.",
            ],
        },
        disc={
            "math": [
                "Solve the equation 3x + 7 = 22 and show your steps.",
                "What is the derivative of f(x) = x³ + 2x² - 5x + 1?",
                "Calculate the area of a circle with radius r = 5 cm.",
                "Prove that the sum of angles in a triangle is 180 degrees.",
                "What is the value of sin(45°) × cos(45°)?",
                "Find the eigenvalues of a 2×2 identity matrix.",
                "Solve the system of equations: 2x + y = 10, x - y = 1.",
                "What is the limit of (sin x)/x as x approaches 0?",
                "Calculate the determinant of [[1,2],[3,4]].",
                "What is the probability of rolling a sum of 7 with two dice?",
                "Compute the integral of ∫ x² dx from 0 to 3.",
                "If a train travels at 80 km/h for 2.5 hours, how far does it go?",
                "What is the logarithm of 1000 base 10?",
                "Find the roots of the quadratic equation x² - 5x + 6 = 0.",
                "Calculate the volume of a sphere with radius r = 3.",
            ],
            "physics": [
                "A 2 kg object is accelerating at 3 m/s². Calculate the force applied.",
                "Explain the difference between kinetic and potential energy.",
                "What is the wavelength of light with frequency 5×10¹⁴ Hz?",
                "Describe Newton's three laws of motion.",
                "Calculate the electric field at a point 2 m from a 5 μC charge.",
                "What happens to the resistance of a wire when its length doubles?",
                "Explain the concept of conservation of momentum.",
                "A ball is thrown upward at 20 m/s. How high does it go?",
                "What is the relationship between pressure and volume in Boyle's Law?",
                "Describe how a transformer works.",
            ],
            "logic": [
                "If all A are B, and all B are C, what can we conclude about A and C?",
                "Prove by contradiction: There is no largest prime number.",
                "Evaluate the logical expression: (P → Q) ∧ P → Q.",
                "If it rains, the ground gets wet. The ground is dry. Did it rain?",
                "Identify the fallacy: 'Everyone I know likes this movie, so it must be good.'",
                "Construct a truth table for (A ∧ B) → (A ∨ C).",
                "Is the statement 'This statement is false' a paradox? Explain.",
                "Use mathematical induction to prove: 1 + 2 + ... + n = n(n+1)/2.",
                "What is the contrapositive of 'If it is summer, then it is hot'?",
                "Resolve this syllogism: Some birds can fly. All penguins are birds. Can penguins fly?",
            ],
            "history": [
                "What were the main causes of World War I?",
                "Describe the significance of the Renaissance period.",
                "Who was the first emperor of China and what dynasty did he found?",
                "Explain the impact of the Industrial Revolution on society.",
                "What was the Silk Road and why was it important?",
                "Summarize the key events of the French Revolution.",
                "What led to the fall of the Roman Empire?",
                "Describe the Meiji Restoration in Japan.",
            ],
            "geography": [
                "Name the seven continents and their largest countries.",
                "What causes the seasons on Earth?",
                "Describe the water cycle and its stages.",
                "What is the Ring of Fire and where is it located?",
                "Explain how mountains are formed through plate tectonics.",
                "What is the difference between weather and climate?",
                "Name the major ocean currents and their effects.",
            ],
            "literature": [
                "Analyze the theme of ambition in Shakespeare's Macbeth.",
                "What is the narrative structure of 'One Hundred Years of Solitude'?",
                "Discuss the role of symbolism in 'The Great Gatsby'.",
                "Compare the writing styles of Hemingway and Fitzgerald.",
                "What makes a poem a sonnet? Describe its characteristics.",
                "Explain the concept of magical realism with examples.",
            ],
        },
        scen={
            "fc": [
                '{"function":"get_weather","parameters":{"city":"Beijing","unit":"celsius"}}',
                '{"function":"calculate","parameters":{"expression":"sqrt(144) + 5 * 3"}}',
                '{"function":"search_database","parameters":{"query":"renewable energy 2024","limit":5}}',
                '{"function":"send_email","parameters":{"to":"user@example.com","subject":"Meeting","body":"Confirmed"}}',
                '{"function":"translate_text","parameters":{"text":"Hello world","source":"en","target":"zh"}}',
                '{"function":"create_calendar_event","parameters":{"title":"Team Meeting","date":"2026-06-15","time":"14:00","duration":60}}',
                '{"function":"get_stock_price","parameters":{"symbol":"AAPL","period":"1d"}}',
                '{"function":"analyze_sentiment","parameters":{"text":"The product is amazing and works perfectly"}}',
                '{"function":"generate_image","parameters":{"prompt":"sunset over mountains","style":"realistic","size":"1024x768"}}',
                '{"function":"summarize_document","parameters":{"url":"https://example.com/report.pdf","max_length":200}}',
            ],
            "code": [
                "Write a Python function to reverse a linked list.",
                "Implement a binary search algorithm in Python.",
                "Write a recursive function to compute Fibonacci numbers.",
                "Implement a hash table from scratch in Python.",
                "Write a function to check if a string is a palindrome.",
                "Implement the quicksort algorithm.",
                "Write a decorator that measures function execution time.",
                "Parse a JSON string and flatten nested dictionaries.",
                "Write an async function to fetch multiple URLs concurrently.",
                "Implement a simple LRU cache.",
            ],
            "math_reasoning": [
                "A store has a 20% discount on all items. If a shirt costs $45 after discount, what was the original price?",
                "A train leaves Station A at 9:00 AM traveling at 60 mph. Another train leaves Station B at 10:00 AM traveling at 75 mph toward Station A. The stations are 300 miles apart. When do they meet?",
                "If a rectangle's length is increased by 25% and its width decreased by 20%, how does the area change?",
                "Three people can paint a fence in 4 hours. How long would it take 5 people to paint the same fence, assuming they work at the same rate?",
            ],
            "translation": [
                "Translate: 'The early bird catches the worm.' → 中文",
                "Translate: '千里之行，始于足下。' → English",
                "Translate: 'Was du heute kannst besorgen, das verschiebe nicht auf morgen.' → English",
                "Translate: 'L'appétit vient en mangeant.' → 中文",
            ],
            "chat": [
                "Hello! How are you doing today?",
                "What do you think about the future of remote work?",
                "Can you recommend a good book to read?",
                "Tell me an interesting fact about space.",
                "What's your opinion on social media's impact on society?",
                "How do you stay productive during the day?",
                "What are some good habits for a healthy lifestyle?",
            ],
        },
    )
