import streamlit as st
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Class descriptions ──
# Used Ai for generation of descriptions #
CLASS_DESCRIPTIONS = {
    "5 alpha- reductase inhibitor (5ARI)": (
        "Blocks an enzyme that converts testosterone into a more potent form (DHT). "
        "Used to shrink enlarged prostates and treat male-pattern hair loss."
    ),
    "5-Nitroimidazole (Antiprotozoal & Antibacterial)": (
        "Kills certain bacteria and parasites by damaging their DNA. "
        "Commonly used for infections like bacterial vaginosis, giardia, and amoebiasis."
    ),
    "Acetylcysteine -Mucolytic": (
        "Breaks down thick, sticky mucus in the airways, making it easier to cough up. "
        "Also used as an antidote for paracetamol (acetaminophen) overdose."
    ),
    "Alkaloids-cytotoxic agents": (
        "Plant-derived compounds that stop cancer cells from dividing by interfering with cell structures. "
        "Used in chemotherapy for various cancers including leukemia and lymphoma."
    ),
    "Alkylating agent": (
        "Attaches chemical groups directly to DNA, preventing cancer cells from replicating. "
        "One of the oldest classes of chemotherapy drugs, used across many cancer types."
    ),
    "Alpha & beta blocker": (
        "Blocks both alpha and beta receptors in the heart and blood vessels, lowering blood pressure. "
        "Used for hypertension and heart failure, especially when other beta blockers are insufficient."
    ),
    "Alpha 2 delta ligands (AED)": (
        "Reduces overactive nerve signals by binding to calcium channels in the nervous system. "
        "Used to treat epilepsy, nerve pain (neuropathy), and anxiety disorders."
    ),
    "Alpha-glucosidase inhibitors": (
        "Slows the breakdown of carbohydrates in the gut, reducing the spike in blood sugar after meals. "
        "Used in type 2 diabetes management as an add-on to other medications."
    ),
    "Amino acids": (
        "The basic building blocks of proteins, given as supplements or intravenous nutrition. "
        "Used when the body cannot get adequate protein from food, such as in severe illness or surgery recovery."
    ),
    "Aminoglycosides": (
        "Antibiotics that kill bacteria by disrupting how they make proteins. "
        "Used for serious gram-negative bacterial infections, often given by injection in hospital settings."
    ),
    "Anabolic steroid": (
        "Synthetic versions of testosterone that promote muscle growth and red blood cell production. "
        "Used medically for muscle-wasting conditions and certain types of anemia."
    ),
    "Analgesic & Antipyretic-PCM": (
        "Paracetamol (acetaminophen) — relieves mild to moderate pain and reduces fever. "
        "Acts on the brain's pain and temperature centers; one of the most widely used medicines worldwide."
    ),
    "Angiotensin receptor blockers(ARB)": (
        "Blocks the action of a hormone (angiotensin II) that tightens blood vessels, allowing them to relax. "
        "Used for high blood pressure, heart failure, and kidney protection in diabetes."
    ),
    "Angiotensin-converting enzyme (ACE) inhibitors": (
        "Stops the body from making angiotensin II, a hormone that raises blood pressure. "
        "Used for hypertension, heart failure, and protecting the kidneys in diabetic patients."
    ),
    "Antacids": (
        "Neutralise excess stomach acid directly and quickly to relieve heartburn and indigestion. "
        "They act within minutes but provide only short-term relief."
    ),
    "Anti-Ulcerants": (
        "Protect and heal the lining of the stomach and small intestine. "
        "Used for peptic ulcers, often by coating the ulcer or reducing acid production."
    ),
    "Anticancer-others": (
        "A broad group of cancer-fighting drugs that don't fit neatly into other chemotherapy categories. "
        "They work through various mechanisms to slow or stop tumour growth."
    ),
    "Anticholinergic- centrally acting": (
        "Blocks acetylcholine signals in the brain to reduce tremors and muscle stiffness. "
        "Used mainly in Parkinson's disease and drug-induced movement disorders."
    ),
    "Anticholinergics": (
        "Blocks acetylcholine signals throughout the body to reduce muscle spasms and secretions. "
        "Used for bladder overactivity, COPD, motion sickness, and irritable bowel syndrome."
    ),
    "Antifibrinolytic": (
        "Prevents blood clots from breaking down too early, helping to control bleeding. "
        "Used during surgery, heavy menstrual bleeding, and trauma to reduce blood loss."
    ),
    "Antifungal Others": (
        "A varied group of antifungal drugs that kill or stop the growth of fungi through different mechanisms. "
        "Used for fungal infections that don't respond to or are not suited for standard antifungals."
    ),
    "Antimalarial- Aminoquinolines": (
        "Kill the malaria parasite inside red blood cells by interfering with its waste disposal system. "
        "Includes classic drugs like chloroquine and hydroxychloroquine."
    ),
    "Antimalarial- Artemisinin and derivatives": (
        "Fast-acting antimalarials derived from the sweet wormwood plant, highly effective against the malaria parasite. "
        "Recommended as first-line treatment for severe and drug-resistant malaria."
    ),
    "Antimalarial- others": (
        "Additional antimalarial drugs that work through various mechanisms not covered by the main classes. "
        "Used in combination therapies or for specific types of malaria."
    ),
    "Antimetabolite- Methotrexate": (
        "Interferes with folate metabolism, which cells need to make DNA and divide. "
        "Used in cancer treatment and as an anti-inflammatory in rheumatoid arthritis and psoriasis."
    ),
    "Antimetabolites": (
        "Mimic natural building blocks of DNA/RNA and get incorporated into them, blocking cell replication. "
        "Used widely in chemotherapy for leukemia, breast cancer, and other cancers."
    ),
    "Antimicrotubule agents- Taxanes": (
        "Stabilise the internal skeleton of cells (microtubules) so cancer cells cannot complete division. "
        "Used for breast, ovarian, lung, and other solid tumour cancers."
    ),
    "Antiprotozoal agents": (
        "Kill or inhibit protozoa — single-celled parasites that cause diseases like giardia, leishmaniasis, and toxoplasmosis. "
        "Used for parasitic infections especially in tropical and subtropical regions."
    ),
    "Antiseptics and disinfectants": (
        "Kill or stop the growth of microorganisms on skin surfaces and wounds to prevent infection. "
        "Applied topically; not intended to be taken internally."
    ),
    "Antiviral (Non-HIV) drugs": (
        "Inhibit viruses from replicating inside the body without killing the host cells. "
        "Used for infections like influenza, herpes, hepatitis B and C, and COVID-19."
    ),
    "Aromatase inhibitor": (
        "Blocks an enzyme (aromatase) that converts other hormones into estrogen, lowering estrogen levels. "
        "Used mainly in postmenopausal women with hormone-sensitive breast cancer."
    ),
    "Atypical Antipsychotics": (
        "Regulate dopamine and serotonin signals in the brain to reduce psychotic symptoms with fewer movement side effects than older drugs. "
        "Used for schizophrenia, bipolar disorder, and treatment-resistant depression."
    ),
    "Benzodiazepines": (
        "Enhance the effect of GABA, the brain's main calming chemical, producing sedation and anxiety relief. "
        "Used for anxiety, insomnia, seizures, and muscle relaxation; can be habit-forming with long-term use."
    ),
    "Beta 2 agonist - Uterus": (
        "Relax the muscles of the uterus by stimulating beta-2 receptors, slowing or stopping contractions. "
        "Used to delay premature labour and give time for other treatments to take effect."
    ),
    "Beta blocker- Cardioselective": (
        "Selectively slow the heart rate and reduce its workload by blocking adrenaline receptors mainly in the heart. "
        "Used for high blood pressure, angina, heart failure, and irregular heartbeats."
    ),
    "Beta blocker- Non selective": (
        "Block adrenaline receptors in both the heart and other tissues like the lungs and blood vessels. "
        "Used for hypertension, tremors, migraines, and some anxiety symptoms."
    ),
    "Biguanides": (
        "Reduce glucose production in the liver and improve the body's sensitivity to insulin. "
        "Metformin is the most well-known; it is the first-line medication for type 2 diabetes."
    ),
    "Bone resorption inhibitor -Bisphosphonates": (
        "Slow down the cells (osteoclasts) that break down bone, preserving bone density. "
        "Used to treat and prevent osteoporosis, and to manage bone damage from cancer."
    ),
    "Calcium channel antagonist/ Antihistamine": (
        "Combines blocking of calcium channels in blood vessels with antihistamine action to prevent migraines and treat allergy symptoms. "
        "Also used to stimulate appetite in some patients."
    ),
    "Calcium channel blocker- migraine prevention": (
        "Relax blood vessels in the brain by blocking calcium entry into vessel walls, reducing the frequency of migraines. "
        "Taken daily as a preventive (prophylactic) treatment, not for acute attacks."
    ),
    "Calcium channel blockers- Dihydropyridines (DHP)": (
        "Relax blood vessel walls by blocking calcium, lowering blood pressure and improving blood flow. "
        "Used for hypertension and angina; less effect on heart rate than other calcium blockers."
    ),
    "Calcium channel blockers- Nondihydropyridines": (
        "Block calcium in both blood vessels and the heart, reducing blood pressure and slowing heart rate. "
        "Used for hypertension, angina, and certain heart rhythm problems."
    ),
    "Cell membrane active agent": (
        "Disrupt the outer membrane of bacteria, causing their contents to leak out and the cell to die. "
        "Used for multidrug-resistant bacterial infections often as a last resort."
    ),
    "Cell wall active agent -Carbapenems": (
        "Broad-spectrum antibiotics that destroy bacteria by blocking the construction of their cell wall. "
        "Reserved for severe or resistant infections in hospital settings."
    ),
    "Cell wall active agent -Extended spectrum Penicillin": (
        "Penicillin-type antibiotics with a broader range of bacterial coverage, including gram-negative bacteria. "
        "Often combined with a beta-lactamase inhibitor to prevent bacterial resistance."
    ),
    "Cephalosporins: 1st generation": (
        "Early-generation antibiotics that kill bacteria by disrupting cell wall synthesis, mainly effective against skin and urinary tract bacteria. "
        "Commonly used for skin infections, strep throat, and surgical prophylaxis."
    ),
    "Cephalosporins: 2nd generation": (
        "Broader than 1st generation, covering more gram-negative bacteria while retaining gram-positive coverage. "
        "Used for respiratory infections, sinusitis, and urinary tract infections."
    ),
    "Cephalosporins: 3 generation": (
        "Further expanded spectrum, particularly effective against gram-negative bacteria including those resistant to earlier cephalosporins. "
        "Used for serious infections like meningitis, pneumonia, and sepsis."
    ),
    "Cephalosporins: 4th generation": (
        "The broadest cephalosporin spectrum, covering both gram-positive and gram-negative bacteria including some resistant strains. "
        "Reserved for severe hospital-acquired infections and febrile neutropenia."
    ),
    "Chloramphenicol": (
        "Stops bacterial protein synthesis by binding to the ribosome; a broad-spectrum antibiotic. "
        "Use is now limited due to serious side effects; reserved for typhoid and certain eye infections."
    ),
    "Cholinesterase inhibitors - Alzheimer's disease": (
        "Prevent the breakdown of acetylcholine in the brain, boosting a chemical messenger involved in memory and thinking. "
        "Used to slow cognitive decline in mild to moderate Alzheimer's disease."
    ),
    "Crystalloids": (
        "Salt-based intravenous fluids that closely mimic body fluid composition to restore hydration and electrolyte balance. "
        "The standard fluids used in hospitals for dehydration, surgery, and shock."
    ),
    "DPP-4 inhibitors": (
        "Block an enzyme that destroys incretin hormones, allowing those hormones to stimulate insulin release after meals. "
        "Used for type 2 diabetes; cause minimal hypoglycemia compared to older diabetes drugs."
    ),
    "Direct antispasmodic (Bladder)": (
        "Relax the muscles of the bladder directly to reduce urgency, frequency, and involuntary leakage of urine. "
        "Used for overactive bladder syndrome."
    ),
    "Disease Modifying Anti-Rheumatoid Drugs (DMARDs)- Non biologics": (
        "Slow or stop the underlying immune attack on joints in rheumatoid arthritis, preventing long-term damage. "
        "Unlike painkillers, they target the disease process itself rather than just symptoms."
    ),
    "Dopamine (D2) receptor antagonist-Prokinetic agent": (
        "Block dopamine receptors in the gut and brain to speed up stomach emptying and reduce nausea and vomiting. "
        "Used for gastroparesis, nausea, and gastroesophageal reflux."
    ),
    "Ectoparasiticides": (
        "Kill parasites that live on the surface of the body, such as lice, scabies mites, and ticks. "
        "Applied topically as creams, lotions, or shampoos."
    ),
    "Erythropoiesis-stimulating agent (ESA)": (
        "Mimic the hormone erythropoietin to stimulate the bone marrow to produce more red blood cells. "
        "Used for anemia caused by chronic kidney disease or chemotherapy."
    ),
    "Estrogens": (
        "Female sex hormones that regulate the reproductive system and are used in hormone replacement therapy. "
        "Also used in contraception and to manage menopausal symptoms like hot flashes."
    ),
    "Fungal ergosterol synthesis inhibitor": (
        "Block an enzyme needed to make ergosterol, the key component of fungal cell membranes, causing the fungus to die. "
        "Used for fungal infections like candidiasis, ringworm, and athlete's foot."
    ),
    "Glucocorticoids": (
        "Powerful anti-inflammatory and immunosuppressant steroids that mimic the body's natural cortisol hormone. "
        "Used for a wide range of conditions including asthma, allergies, arthritis, and autoimmune diseases."
    ),
    "Glycopeptides": (
        "Antibiotics that block bacterial cell wall synthesis through a different mechanism than beta-lactams. "
        "Used for serious gram-positive infections including MRSA, typically given intravenously."
    ),
    "Gonadotropins": (
        "Hormones that stimulate the ovaries or testes to produce sex hormones and eggs or sperm. "
        "Used in fertility treatments to stimulate ovulation or sperm production."
    ),
    "H1 Antihistaminics (First Generation)": (
        "Block histamine receptors to relieve allergy symptoms, but also cause significant drowsiness due to brain penetration. "
        "Used for allergies, hay fever, motion sickness, and as sleep aids."
    ),
    "H1 Antihistaminics (second Generation)": (
        "Block histamine receptors for allergy relief with much less drowsiness than first-generation drugs. "
        "The standard modern antihistamines for hay fever, hives, and other allergic reactions."
    ),
    "H2 Receptor Blocker": (
        "Block histamine H2 receptors in the stomach lining, reducing the amount of acid produced. "
        "Used for heartburn, peptic ulcers, and gastroesophageal reflux disease (GERD)."
    ),
    "HMG CoA inhibitors (statins)": (
        "Block the liver enzyme responsible for producing cholesterol, significantly lowering LDL (bad) cholesterol levels. "
        "Used to reduce the risk of heart attacks and strokes in people with high cholesterol or cardiovascular disease."
    ),
    "Haemopoetic agents": (
        "Stimulate the bone marrow to produce more blood cells — red cells, white cells, or platelets. "
        "Used for various types of anemia and to support recovery after chemotherapy."
    ),
    "Haemostatic agent": (
        "Help stop bleeding by promoting blood clotting or strengthening blood vessels. "
        "Used for excessive bleeding during surgery, trauma, or bleeding disorders."
    ),
    "Hepatoprotectives": (
        "Protect liver cells from damage caused by toxins, alcohol, drugs, or disease. "
        "Used as supportive therapy in liver conditions like hepatitis and fatty liver disease."
    ),
    "High-ceiling Diuretics (Inhibitors of Na+-K+- 2Cl cotransport)": (
        "Powerful diuretics (loop diuretics like furosemide) that cause the kidneys to excrete large amounts of water and salt. "
        "Used for fluid overload in heart failure, kidney disease, and liver cirrhosis."
    ),
    "Histamine analog- Meniere's Disease": (
        "Mimics histamine in the inner ear's blood vessels, improving fluid balance and blood flow. "
        "Used to reduce the frequency of vertigo attacks in Meniere's disease."
    ),
    "Immunosuppressant- Calcineurin inhibitors": (
        "Suppress the immune system by blocking a key signalling molecule (calcineurin) that activates immune cells. "
        "Used to prevent organ rejection after transplants and for autoimmune skin conditions."
    ),
    "Immunosuppressant- Purine analogs": (
        "Block the production of DNA building blocks in rapidly dividing immune cells, dampening the immune response. "
        "Used to prevent organ rejection and treat autoimmune diseases like lupus and inflammatory bowel disease."
    ),
    "Leukotriene antagonists": (
        "Block leukotrienes — inflammatory chemicals released during allergic reactions and asthma attacks. "
        "Used as maintenance therapy for asthma and to relieve allergic rhinitis symptoms."
    ),
    "Lincosamides": (
        "Antibiotics that stop bacteria from making proteins, effective mainly against gram-positive bacteria and anaerobes. "
        "Commonly used for skin infections, bone infections, and as an alternative to penicillin in allergic patients."
    ),
    "Lipase inhibitor": (
        "Block the enzyme in the gut that digests fat, reducing how much dietary fat is absorbed into the body. "
        "Used as part of a weight loss programme for obesity management."
    ),
    "Local anaesthetic (Amides)": (
        "Block sodium channels in nerve fibres, preventing pain signals from reaching the brain. "
        "Used for numbing during surgical procedures, dental work, and pain management."
    ),
    "Low molecular weight Heparin (LMWH)": (
        "Prevent blood clots from forming or growing by inhibiting clotting factors in the coagulation cascade. "
        "Used for deep vein thrombosis, pulmonary embolism prevention, and during surgery."
    ),
    "Low-ceiling Diuretics (Inhibitors of Na+Cl symport)": (
        "Thiazide diuretics that cause moderate removal of water and salt via the kidneys. "
        "Used for high blood pressure and mild fluid retention; their effect plateaus at higher doses."
    ),
    "Macrolides": (
        "Antibiotics that halt bacterial growth by blocking protein synthesis inside the bacterial ribosome. "
        "Used for respiratory infections, atypical pneumonia, sexually transmitted infections, and as a penicillin alternative."
    ),
    "Mood stabilizers": (
        "Stabilise mood swings between mania and depression in bipolar disorder. "
        "Used long-term to prevent both manic and depressive episodes; the classic example is lithium."
    ),
    "Mycobacterial cell wall inhibitors-Ethambutol": (
        "Specifically targets and disrupts the construction of the mycobacterial cell wall. "
        "Used as part of combination therapy for tuberculosis and other mycobacterial infections."
    ),
    "NO Donors": (
        "Release nitric oxide in the body, which relaxes and widens blood vessels to improve blood flow. "
        "Used for angina (chest pain) and acute heart failure; nitroglycerin is the classic example."
    ),
    "NSAID's -Selective COX-2 Inhibitors": (
        "Target only the COX-2 enzyme that produces pain and inflammation, sparing the stomach-protective COX-1. "
        "Used for arthritis and pain with a lower risk of stomach ulcers than traditional NSAIDs."
    ),
    "NSAID's- Non-Selective COX 1&2 Inhibitors (Topical)": (
        "Applied directly to the skin to reduce local pain and inflammation with minimal systemic side effects. "
        "Used for muscle pain, joint pain, and sports injuries."
    ),
    "NSAID's- Non-Selective COX 1&2 Inhibitors (acetic acid)": (
        "Acetic acid-derived NSAIDs (like diclofenac, indomethacin) that block both COX enzymes to reduce pain and inflammation. "
        "Used for arthritis, acute pain, and inflammatory conditions."
    ),
    "NSAID's- Non-Selective COX 1&2 Inhibitors (enolic acids)": (
        "Enolic acid-derived NSAIDs (like piroxicam, meloxicam) with long half-lives allowing once-daily dosing. "
        "Used for chronic conditions like osteoarthritis and rheumatoid arthritis."
    ),
    "NSAID's- Non-Selective COX 1&2 Inhibitors (fenamates)": (
        "Fenamate NSAIDs (like mefenamic acid) that block COX enzymes and have some additional anti-inflammatory properties. "
        "Commonly used for period pain (dysmenorrhoea) and mild to moderate pain."
    ),
    "NSAID's- Non-Selective COX 1&2 Inhibitors (propionic acid)": (
        "Propionic acid-derived NSAIDs (like ibuprofen, naproxen) — among the most widely used over-the-counter pain relievers. "
        "Used for headaches, fever, period pain, and inflammatory conditions."
    ),
    "NSAID's-Non-Selective COX 1&2 Inhibitors (Others)": (
        "Other NSAIDs not fitting the standard chemical subgroups, still blocking COX enzymes to relieve pain and inflammation. "
        "Used for various pain and inflammatory conditions."
    ),
    "Natural Progesterone": (
        "Provides progesterone, a hormone that supports pregnancy and regulates the menstrual cycle. "
        "Used in hormone replacement therapy, fertility treatments, and to support early pregnancy."
    ),
    "Non-benzodiazepine hypnotics (Z Compounds)": (
        "Act on the same brain receptors as benzodiazepines to promote sleep but with a more targeted effect and shorter duration. "
        "Used for short-term insomnia; generally considered safer than benzodiazepines for sleep."
    ),
    "Nootropic agent": (
        "Intended to enhance cognitive functions like memory, focus, and mental clarity. "
        "Used for cognitive decline and conditions affecting brain function, though evidence varies widely."
    ),
    "Noradrenergic and specific serotonergic antidepressants (NASSAs)": (
        "Boost both noradrenaline and serotonin levels in the brain through a unique mechanism, with sedating properties. "
        "Used for depression, especially when insomnia or poor appetite are prominent symptoms."
    ),
    "Nucleoside/nucleotide reverse transcriptase inhibitors (NRTIs) ": (
        "Block the enzyme HIV uses to copy itself, preventing the virus from replicating inside the body. "
        "A cornerstone of antiretroviral therapy (ART) for HIV infection."
    ),
    "Opioids": (
        "Bind to opioid receptors in the brain and spinal cord to powerfully reduce the perception of pain. "
        "Used for moderate to severe pain; carry a risk of dependence and respiratory depression with misuse."
    ),
    "Oral Factor Xa Inhibitors": (
        "Block Factor Xa, a key protein in the blood clotting cascade, to prevent harmful clot formation. "
        "Used to prevent and treat deep vein thrombosis, pulmonary embolism, and stroke in atrial fibrillation."
    ),
    "Osmotic laxatives/Purgative": (
        "Draw water into the bowel from surrounding tissues, softening stool and stimulating bowel movement. "
        "Used for constipation and to clear the bowel before colonoscopy or surgery."
    ),
    "Osteoarthritis- Hyaluronic acid": (
        "Injected into joints to supplement the natural lubricating fluid, reducing friction and pain. "
        "Used as a viscosupplementation therapy for knee osteoarthritis when other treatments have failed."
    ),
    "Oxazolidinone": (
        "A newer class of antibiotics that block bacterial protein synthesis at a very early stage. "
        "Reserved for serious gram-positive infections resistant to other antibiotics, including MRSA and VRE."
    ),
    "P2Y12 inhibitors (ADP receptor)": (
        "Block platelet activation by inhibiting a receptor (P2Y12) that triggers clumping, preventing blood clots in arteries. "
        "Used after heart attacks, stents, and in atrial fibrillation to reduce stroke risk."
    ),
    "Partial estrogen agonist": (
        "Acts like estrogen in some tissues (like bone) while blocking it in others (like breast tissue). "
        "Used to treat and prevent osteoporosis and reduce breast cancer risk in postmenopausal women."
    ),
    "Penem": (
        "Beta-lactam antibiotics with an exceptionally broad spectrum including MRSA and resistant gram-negative bacteria. "
        "Used as a last-resort antibiotic for multidrug-resistant infections."
    ),
    "Phosphodiesterase-IV inhibitors (Smooth Muscle Relaxant)": (
        "Block an enzyme (PDE4) that causes airway inflammation and constriction, reducing flare-ups. "
        "Used for severe COPD and psoriatic arthritis to reduce inflammation."
    ),
    "Phosphodiesterase-V inhibitors": (
        "Relax blood vessels in the penis and lungs by blocking PDE5, an enzyme that restricts blood flow. "
        "Used for erectile dysfunction and pulmonary arterial hypertension."
    ),
    "Phosphorous binder": (
        "Bind dietary phosphorus in the gut so it cannot be absorbed into the bloodstream. "
        "Used in chronic kidney disease where the kidneys can no longer filter out excess phosphorus."
    ),
    "Platinum compounds-Anticancer": (
        "Form cross-links within cancer cell DNA, blocking replication and triggering cell death. "
        "Used in chemotherapy for testicular, ovarian, bladder, and lung cancers."
    ),
    "Potassium channel opener": (
        "Open potassium channels in blood vessel walls, causing them to relax and widen to lower blood pressure. "
        "Used for severe hypertension resistant to other treatments."
    ),
    "Progestins (First generation)": (
        "Synthetic forms of progesterone used in contraception and hormone therapy. "
        "Used in combined or progestin-only contraceptive pills and hormone replacement therapy."
    ),
    "Protein synthesis inhibitors": (
        "Block the bacterial machinery (ribosome) used to make proteins, halting bacterial growth or killing bacteria. "
        "A broad category covering several antibiotic classes including macrolides, tetracyclines, and aminoglycosides."
    ),
    "Proteoglycan synthesis Stimulator": (
        "Stimulate the production of cartilage components to support joint repair and reduce degeneration. "
        "Used as supportive therapy in osteoarthritis to slow cartilage breakdown."
    ),
    "Proteolytic Enzymes": (
        "Break down proteins involved in inflammation, swelling, and clot formation to promote healing. "
        "Used as anti-inflammatory agents after surgery, injuries, and to aid tissue repair."
    ),
    "Proton pump inhibitors": (
        "Irreversibly block the acid pump in the stomach lining, providing powerful and long-lasting acid suppression. "
        "Used for GERD, peptic ulcers, H. pylori eradication, and to protect the stomach during NSAID use."
    ),
    "Quinolones/ Fluroquinolones": (
        "Broad-spectrum antibiotics that kill bacteria by blocking enzymes (DNA gyrase and topoisomerase IV) needed to copy DNA. "
        "Used for urinary tract, respiratory, and gastrointestinal infections; avoid in children due to joint effects."
    ),
    "RNA polymerase inhibitors- Rifamycins": (
        "Block the bacterial enzyme that copies RNA, preventing bacteria from making the proteins they need to survive. "
        "Used as a key component of tuberculosis treatment and for some other serious bacterial infections."
    ),
    "Retinoids- First generation": (
        "Vitamin A derivatives that regulate skin cell growth and turnover, reducing abnormal skin thickening. "
        "Used for severe acne, psoriasis, and certain skin disorders; must be avoided in pregnancy."
    ),
    "SGLT2 inhibitors": (
        "Block a protein in the kidneys (SGLT2) that reabsorbs glucose, causing excess sugar to be excreted in urine. "
        "Used for type 2 diabetes; also shown to protect the heart and kidneys independently of blood sugar control."
    ),
    "Selective Seretonin Reuptake inhibitors (SSRIs)": (
        "Increase serotonin levels in the brain by blocking its reabsorption, improving mood and reducing anxiety. "
        "The most commonly prescribed antidepressants; also used for OCD, PTSD, and panic disorder."
    ),
    "Serotonin (5-HT4) receptor agonist-Prokinetic agent": (
        "Stimulate serotonin receptors in the gut to speed up bowel movement and stomach emptying. "
        "Used for constipation-predominant irritable bowel syndrome and gastroparesis."
    ),
    "Serotonin antagonists (5-HT3 antagonists)": (
        "Block serotonin receptors in the gut and brain that trigger the vomiting reflex. "
        "Highly effective antiemetics used to prevent nausea and vomiting from chemotherapy and surgery."
    ),
    "Serotonin-norepinephrine reuptake inhibitors (SNRIs)": (
        "Raise both serotonin and norepinephrine levels in the brain, improving mood and reducing pain signals. "
        "Used for depression, anxiety disorders, and chronic pain conditions like fibromyalgia."
    ),
    "Short acting β2-agonists": (
        "Rapidly relax the muscles around the airways by stimulating beta-2 receptors, opening up constricted bronchial passages. "
        "The go-to 'rescue inhaler' for immediate relief of asthma and COPD attacks."
    ),
    "Skeletal muscle relaxant- Centrally acting": (
        "Reduce muscle spasms and stiffness by acting on the central nervous system rather than the muscles directly. "
        "Used for muscle spasms, back pain, and spasticity from neurological conditions."
    ),
    "Sodium channel modulators (AED)": (
        "Stabilise overactive neurons by blocking sodium channels, preventing the rapid, repetitive firing that causes seizures. "
        "Used to treat epilepsy and also for nerve pain and bipolar disorder."
    ),
    "Stimulant laxatives/purgative": (
        "Irritate the lining of the bowel to stimulate muscle contractions and speed up transit of stool. "
        "Used for constipation; intended for short-term use as they can cause dependence with prolonged use."
    ),
    "Stomatologicals- Antimicrobials": (
        "Antimicrobial agents formulated specifically for the mouth to treat oral infections and inflammation. "
        "Used for gum disease, mouth ulcers, and oral thrush."
    ),
    "Sulfonylureas (Insulin Secretogogues)": (
        "Stimulate the pancreas to release more insulin regardless of blood sugar levels. "
        "Used for type 2 diabetes; effective but carry a risk of low blood sugar (hypoglycemia)."
    ),
    "Sympthatomimmetics- alpha 1 (Nasal)": (
        "Constrict blood vessels in the nasal passages by stimulating alpha-1 receptors, reducing swelling and congestion. "
        "Used as nasal decongestants for colds and allergic rhinitis; typically short-term use only."
    ),
    "Synaptic vescicle 2 A protein ligand (AED)": (
        "Bind to a protein (SV2A) involved in releasing neurotransmitters from nerve endings, reducing abnormal electrical activity. "
        "Used as an anticonvulsant for partial-onset and generalised seizures; levetiracetam is the key example."
    ),
    "Tear substitues": (
        "Lubricate and moisturise the surface of the eye to relieve dryness and irritation. "
        "Used for dry eye syndrome; available as drops, gels, and ointments."
    ),
    "Tetracyclines": (
        "Broad-spectrum antibiotics that block protein synthesis in bacteria by binding to the ribosome. "
        "Used for acne, Lyme disease, chlamydia, and atypical pneumonia."
    ),
    "Theophylline & its derivatives": (
        "Relax airway smooth muscle and reduce inflammation to open up the airways, while also mildly stimulating breathing. "
        "Used for asthma and COPD; require careful dosing as the therapeutic window is narrow."
    ),
    "Thiazolidinedione(PPAR gamma agonist)": (
        "Improve insulin sensitivity throughout the body by activating a nuclear receptor (PPAR-gamma) involved in fat and glucose metabolism. "
        "Used for type 2 diabetes, particularly in patients with significant insulin resistance."
    ),
    "Thyroid hormones": (
        "Supplement or replace the hormones produced by the thyroid gland to regulate metabolism. "
        "Used for hypothyroidism (underactive thyroid) and after thyroid removal."
    ),
    "Tricyclic antidepressants": (
        "Increase noradrenaline and serotonin levels in the brain by blocking their reuptake; one of the older antidepressant classes. "
        "Used for depression, nerve pain, migraines, and bedwetting in children."
    ),
    "Typical Antipsychotics": (
        "Block dopamine receptors in the brain to reduce psychotic symptoms like hallucinations and delusions. "
        "The original antipsychotic class; effective but more prone to causing movement side effects than newer drugs."
    ),
    "Tyrosine kinase inhibitors": (
        "Block specific enzymes (tyrosine kinases) that cancer cells use to grow and survive. "
        "Targeted therapy used for specific cancers like CML, lung cancer, and breast cancer driven by known genetic mutations."
    ),
    "Uricosuric agent-gout": (
        "Increase the kidney's excretion of uric acid, lowering its levels in the blood and preventing gout attacks. "
        "Used for chronic gout in patients who over-retain uric acid rather than over-produce it."
    ),
    "Urinary Antiseptic-Nitrofurantoin": (
        "Concentrates in the urine where it damages multiple bacterial targets simultaneously, killing urinary tract bacteria. "
        "Used specifically for uncomplicated urinary tract infections (UTIs); not suitable for kidney infections."
    ),
    "Uroselective adrenergic receptor(α1a) antagonist": (
        "Selectively relax the smooth muscle of the prostate and bladder neck to improve urine flow. "
        "Used for symptoms of benign prostatic hyperplasia (enlarged prostate) like difficulty urinating."
    ),
    "Uterotonic and abortificient": (
        "Stimulate uterine muscle contractions, either to induce labour, manage postpartum bleeding, or end a pregnancy. "
        "Used in obstetric settings under close medical supervision."
    ),
    "Vitamins": (
        "Essential micronutrients that the body requires in small amounts for normal metabolic functions. "
        "Used to treat or prevent deficiency states; supplemented when dietary intake or absorption is insufficient."
    ),
    "Xanthine oxidase Inhibitors-gout": (
        "Block the enzyme that produces uric acid, lowering its levels in the blood and preventing gout and kidney stones. "
        "The most commonly used long-term treatment for chronic gout; allopurinol is the classic example."
    ),
    "μ-opioid receptor & norepinephrine reuptake inhibitor (NRI)": (
        "Combine opioid pain relief with norepinephrine reuptake inhibition, providing pain control through two complementary pathways. "
        "Used for moderate to severe chronic pain, particularly musculoskeletal pain."
    ),
}


@st.cache_resource
def load_components():
    
    hf_repo_name = "playtozon/drug-action-classifier"
    
    tokenizer = AutoTokenizer.from_pretrained(hf_repo_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_repo_name)
    
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return tokenizer, model, le

tokenizer, model, le = load_components()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# UI
st.title(" -- Drug Action Class Predictor --")
st.write("Enter the drug details below to predict its action class.")

drug_name = st.text_input("Drug Name (e.g., amoxicillin 500mg capsule)", "")
uses = st.text_area("Uses (e.g., Treatment of bacterial infections)", "")
side_effects = st.text_area("Side Effects (e.g., Nausea, diarrhea)", "")

if st.button("Predict Action Class"):
    if not drug_name and not uses and not side_effects:
        st.warning("Are you on Drugs ?! Enter valid details for prediction. ")
    else:
        input_text = f"Drug: {drug_name}. Uses: {uses}. Side Effects: {side_effects}"

        encoding = tokenizer(
            input_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with st.spinner("Analyzing..."):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[0]
                top3 = torch.topk(probs, k=3)
                top3_indices = top3.indices.cpu().numpy()
                top3_probs = top3.values.cpu().numpy()

            top3_classes = le.inverse_transform(top3_indices)

        # Results
        st.markdown("### 🔬 Predicted Action Classes")
        st.error("As this is a small scale model, actual results may vary.")
        st.success("The top 3 most likely Action Classes are displayed.")
        medals = ["🥇", "🥈", "🥉"]

        for i, cls in enumerate(top3_classes):
            cls_stripped  = cls.strip()
            description   = CLASS_DESCRIPTIONS.get(
                cls_stripped,
                "No description available for this class."
            )

            with st.expander(f"{medals[i]}  {cls_stripped}", expanded=True):
                st.markdown(f"_{description}_")