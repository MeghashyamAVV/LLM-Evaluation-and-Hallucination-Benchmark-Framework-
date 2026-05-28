from evaluator import EvalSample

EVAL_DATASET = [
    EvalSample(
        question="What is the powerhouse of the cell?",
        context="The mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells. They generate most of the cell's supply of ATP, used as a source of chemical energy. They are often referred to as the powerhouse of the cell.",
        ground_truth="The mitochondria is the powerhouse of the cell. It generates most of the cell's ATP energy."
    ),
    EvalSample(
        question="When did World War II end?",
        context="World War II was a global conflict that lasted from 1939 to 1945. It ended in Europe on May 8, 1945, known as Victory in Europe Day (V-E Day). The war in the Pacific ended on September 2, 1945, when Japan formally surrendered aboard the USS Missouri.",
        ground_truth="World War II ended in 1945. In Europe it ended on May 8, 1945 (V-E Day), and in the Pacific on September 2, 1945 when Japan surrendered."
    ),
    EvalSample(
        question="What causes the seasons on Earth?",
        context="Earth experiences seasons because its axis is tilted at approximately 23.5 degrees relative to its orbital plane around the sun. When the Northern Hemisphere is tilted toward the sun, it experiences summer. When it is tilted away, it experiences winter. This is not caused by Earth's distance from the sun.",
        ground_truth="Seasons are caused by Earth's axial tilt of 23.5 degrees, not its distance from the sun."
    ),
    EvalSample(
        question="What is machine learning?",
        context="Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves. The process begins with observations or data to look for patterns.",
        ground_truth="Machine learning is a branch of AI where systems learn from data and experience without being explicitly programmed."
    ),
    EvalSample(
        question="How does photosynthesis work?",
        context="Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy stored in glucose. It occurs mainly in the chloroplasts. The overall equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. Plants absorb sunlight using chlorophyll.",
        ground_truth="Photosynthesis converts light energy into chemical energy (glucose) using CO2, water, and sunlight. It occurs in chloroplasts."
    ),
    EvalSample(
        question="What is the speed of light?",
        context="The speed of light in a vacuum is approximately 299,792,458 meters per second, often approximated as 3 × 10^8 m/s or 300,000 km/s. It is denoted by the letter c and is a fundamental constant in physics. Nothing with mass can travel at or faster than the speed of light.",
        ground_truth="The speed of light in vacuum is approximately 299,792,458 meters per second (about 300,000 km/s)."
    ),
    EvalSample(
        question="Who invented the telephone?",
        context="The telephone was invented by Alexander Graham Bell, who was awarded the first patent for it on March 7, 1876. Bell made the first successful telephone call on March 10, 1876, to his assistant Thomas Watson. Elisha Gray also developed a similar device around the same time.",
        ground_truth="Alexander Graham Bell invented the telephone and received the patent on March 7, 1876."
    ),
    EvalSample(
        question="What is DNA?",
        context="DNA, or deoxyribonucleic acid, is a molecule that carries the genetic instructions for the development, functioning, growth, and reproduction of all known organisms. DNA is made up of two strands that wind around each other to form a double helix. Each strand is made of four chemical bases: adenine (A), guanine (G), cytosine (C), and thymine (T).",
        ground_truth="DNA (deoxyribonucleic acid) is a double-helix molecule made of four bases (A, G, C, T) that carries genetic instructions."
    ),
    EvalSample(
        question="What is the capital of Australia?",
        context="Australia is a country and continent. Its capital city is Canberra, which was purpose-built to serve as the nation's capital. Many people mistakenly believe Sydney or Melbourne is the capital, but Canberra was chosen as a compromise between those two rival cities when Australia became a federation in 1901.",
        ground_truth="The capital of Australia is Canberra, not Sydney or Melbourne."
    ),
    EvalSample(
        question="How does the immune system fight viruses?",
        context="When a virus enters the body, the immune system responds in two ways. The innate immune response provides immediate non-specific defense, triggering inflammation. The adaptive immune response is slower but more specific: B cells produce antibodies that neutralize the virus, while T cells destroy infected cells. Memory cells are created so future responses are faster.",
        ground_truth="The immune system fights viruses via innate (immediate, non-specific) and adaptive (specific) responses. B cells make antibodies, T cells destroy infected cells."
    ),
    EvalSample(
        question="What is blockchain technology?",
        context="A blockchain is a distributed digital ledger that records transactions across many computers so that records cannot be altered retroactively. Each block contains transaction data, a timestamp, and a cryptographic hash of the previous block, forming a chain. It was originally invented for Bitcoin but has many other applications.",
        ground_truth="Blockchain is a distributed ledger where data is stored in linked blocks secured by cryptographic hashes, making records tamper-resistant."
    ),
    EvalSample(
        question="What causes earthquakes?",
        context="Earthquakes are caused by the sudden release of energy in Earth's crust, creating seismic waves. Most earthquakes occur along fault lines where tectonic plates meet. When stress builds up along these faults and is suddenly released, the ground shakes. The point underground where the earthquake originates is called the focus or hypocenter.",
        ground_truth="Earthquakes are caused by the sudden release of energy along fault lines where tectonic plates meet, creating seismic waves."
    ),
    EvalSample(
        question="What is the water cycle?",
        context="The water cycle, or hydrological cycle, describes the continuous movement of water on, above, and below Earth's surface. Key processes include: evaporation (water turns to vapor), condensation (vapor forms clouds), precipitation (rain or snow falls), and collection (water gathers in oceans, rivers, and groundwater).",
        ground_truth="The water cycle is the continuous movement of water through evaporation, condensation, precipitation, and collection."
    ),
    EvalSample(
        question="What is the difference between RAM and storage?",
        context="RAM (Random Access Memory) is short-term memory used by your computer to store data it is actively using. It is fast but loses data when powered off. Storage (like an SSD or HDD) is long-term memory that keeps data permanently. More RAM means your computer can handle more tasks simultaneously; more storage means more files can be saved.",
        ground_truth="RAM is fast short-term memory for active tasks that clears when powered off. Storage is permanent long-term memory for files."
    ),
    EvalSample(
        question="How do vaccines work?",
        context="Vaccines work by training the immune system to recognize and fight specific pathogens. They contain weakened, killed, or partial versions of a virus or bacteria, or genetic instructions (mRNA) to make a recognizable protein. When the immune system encounters this, it creates memory cells so it can respond much faster if exposed to the real pathogen later.",
        ground_truth="Vaccines train the immune system by exposing it to harmless versions of a pathogen, prompting memory cell creation for future protection."
    ),
    EvalSample(
        question="What is gravity?",
        context="Gravity is a fundamental force of nature that attracts objects with mass toward each other. On Earth, gravity gives weight to physical objects and causes them to fall when dropped. Isaac Newton described gravity as a force proportional to mass and inversely proportional to distance squared. Einstein later described it as a curvature of spacetime.",
        ground_truth="Gravity is the fundamental force attracting objects with mass toward each other, described by Newton as proportional to mass and by Einstein as spacetime curvature."
    ),
    EvalSample(
        question="What is inflation in economics?",
        context="Inflation is the rate at which the general level of prices for goods and services rises over time, eroding purchasing power. It is measured by the Consumer Price Index (CPI). Moderate inflation is considered normal in a healthy economy. Central banks, like the Federal Reserve, use interest rates to control inflation.",
        ground_truth="Inflation is the rate of price increases over time, measured by CPI, which erodes purchasing power. Central banks control it via interest rates."
    ),
    EvalSample(
        question="What is the greenhouse effect?",
        context="The greenhouse effect is a natural process that warms Earth's surface. Sunlight passes through the atmosphere and warms the ground. The ground then emits heat, which is absorbed by greenhouse gases like CO2, methane, and water vapor, trapping it in the atmosphere. Human activities have enhanced this effect, leading to climate change.",
        ground_truth="The greenhouse effect is when greenhouse gases like CO2 trap heat from Earth's surface, warming the planet. Human activities have intensified this."
    ),
    EvalSample(
        question="How does the internet work?",
        context="The internet is a global network of computers connected by cables, wireless signals, and satellites. Data is broken into small packets and sent via routers across networks using the TCP/IP protocol suite. Each device has an IP address. The World Wide Web (websites) is one service that runs on the internet using HTTP protocol.",
        ground_truth="The internet connects computers globally via TCP/IP protocols. Data is sent as packets through routers. The Web is one service running on it using HTTP."
    ),
    EvalSample(
        question="What is Newton's second law of motion?",
        context="Newton's second law of motion states that the acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass. It is expressed as the equation F = ma, where F is force in Newtons, m is mass in kilograms, and a is acceleration in meters per second squared.",
        ground_truth="Newton's second law states F = ma: force equals mass times acceleration."
    ),
]