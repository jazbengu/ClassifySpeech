# ClassifySpeech

**Real-Time Speech-to-Speech Translation with Spatial Audio for Multilingual Smart Learning Environments**

[![Python](https://img.shields.io/badge/Python-3.12.4-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.x-green.svg)](https://www.djangoproject.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

**ClassifySpeech** is a prototype Smart Learning Environment (SLE) that integrates real-time Speech-to-Speech Translation (S2ST) with spatial audio technology to facilitate cross-lingual communication in multilingual classrooms. Built as part of a research project investigating how perspective sharing can support translanguaging, this prototype addresses the critical gap in current SLEs: the lack of tools that enable students to fully utilise their linguistic repertoire for improved comprehension and engagement.

The system enables students speaking different languages (currently English, isiZulu, and Afrikaans) to communicate naturally through automatic translation, while spatial audio—implemented via Head Related Transfer Function (HRTF)—creates an immersive auditory environment that mimics natural sound localisation.

---

## 🎯 Problem Statement

Current Smart Learning Environments lack effective tools for translanguaging—a pedagogical approach that allows multilingual students to leverage their full linguistic repertoire for improved comprehension. Despite South Africa's rich linguistic diversity, where most students know more than one language, existing SLEs:

- Do not support real-time cross-lingual communication
- Lack perspective sharing mechanisms for immersive collaborative learning
- Fail to accommodate the cultural and cognitive resources of multilingual students

This gap limits the inclusive potential of SLEs and hinders academic success for multilingual learners.

---

## 💡 Solution

ClassifySpeech addresses these challenges through three integrated components:

1. **Real-Time Speech-to-Speech Translation**: Converts spoken language directly into another spoken language using Automatic Speech Recognition (ASR), Neural Machine Translation (NMT), and Text-to-Speech (TTS) synthesis.

2. **Perspective Sharing**: Enables multiple users to view, explore, and interact within a shared virtual environment, fostering presence and connection across geographical distances.

3. **Spatial Audio (HRTF)**: Simulates 3D auditory cues that reflect how natural sound reaches a listener's ears from different directions and distances, enhancing immersion and comprehension.

---

## 🧪 Methodology

The prototype was developed following **Design Science Research (DSR)** principles as conceptualised by Hevner, focusing on iterative creation and refinement of artifacts to solve real-world problems.

### Development Approach

- **Iterative Prototyping**: Each mini-prototype was built, tested, and refined based on user feedback
- **Controlled & Classroom Testing**: Evaluations conducted in both isolated environments and real classroom settings (15-person classroom)
- **User-Centred Design**: Continuous refinement of interface elements, translation accuracy, and audio clarity

### Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Translation Accuracy | ≥ 80% | Percentage of correct translations against references |
| Response Time | < 3 seconds | Delay between audio input and translated output |
| Audio Clarity | High | Consistency and clarity of speech output with spatial audio |

---

## 🏗️ Technical Stack

### Framework & Language
- **Django** (MVT Architecture) - Primary web framework for robust audio handling
- **Python 3.12.4** - Core programming language

### Key Libraries

| Library | Purpose |
|---------|---------|
| `gTTS` (unofficial) | Text-to-speech conversion for natural audio output |
| `googleTrans` | Multilingual text translation |
| `speech_recognition` | Audio-to-text conversion |
| `pydub` | Audio file manipulation (format conversion, segmentation) |
| `noisereduce` | Background noise reduction for clearer input |
| `librosa` | Audio analysis and feature extraction |
| `numpy` & `scipy` | Mathematical operations and audio waveform processing |

### Audio Processing Pipeline

1. **Input**: User speaks into microphone → audio saved as WAV
2. **Noise Reduction**: Background noise minimised for clarity
3. **Speech Recognition**: Audio transcribed to text
4. **Translation**: Text translated to target language
5. **TTS Synthesis**: Translated text converted to speech
6. **Spatial Audio**: HRTF applied for 3D sound localisation

---

## ✨ Features

- **Real-Time Translation**: Supports English, isiZulu, and Afrikaans
- **Spatial Audio**: HRTF-based 3D sound localisation for immersive experience
- **Noise Reduction**: Enhanced audio clarity in classroom environments
- **Intuitive Interface**: User-centred design with written translation display
- **Cross-Platform**: Web-based accessibility

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12.4 or higher
- pip (Python package manager)
- Git
- Working microphone and speakers/headphones

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/jazbengu/ClassifySpeech.git
cd ClassifySpeech
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Apply database migrations**
```bash
python manage.py migrate
```

5. **Run the development server**
```bash
python manage.py runserver
```

6. **Access the application**
Open your browser and navigate to `http://127.0.0.1:8000`

---

## 📊 Results & Findings

### Translation Performance

| Language Pair | Accuracy | Notes |
|---------------|----------|-------|
| English ↔ Afrikaans | High (~80%+) | Strong performance due to library support |
| English ↔ isiZulu | Moderate (~60%) | Limited TTS support for African languages |

### Key Findings

- **European languages** (Afrikaans, French, Spanish) demonstrated strong performance with accurate and natural audio output
- **African languages** (isiZulu) showed reasonable translation accuracy but significant challenges in natural voice synthesis
- **Phonetic challenges**: Bantu languages have tonal nuances not captured by TTS libraries designed for Germanic/Romance languages
- **Spatial audio**: HRTF performed adequately in controlled environments (small room, empty classroom), with optimal results using stereo earphones

---

## ⚠️ Challenges & Limitations

### Technical Limitations

1. **TTS Support**: Limited support for South African languages; Google Translate lacks vocal support for African languages
2. **Library Compatibility**: Mozilla TTS only supports Python 3.3–3.9, incompatible with Python 3.12.4
3. **Cultural Context**: Systems lack awareness of cultural nuances (e.g., greeting elders requires "Sanibonani" vs "Sawubona")
4. **HRTF Customisation**: Finding optimal Head Related Transfer Function parameters is time-consuming; everyone's ears are anatomically unique

### Translation-Specific Issues

- **Tonal accuracy**: isiZulu's tonal nuances often missed in translation
- **Phonetic inaccuracies**: Become more apparent in complex classroom sentences
- **Cost barriers**: Commercial TTS services (Amazon Polly, Google Cloud TTS) require payment

---

## 🔮 Future Work

1. **African Language TTS**: Develop and train synthetic voices specifically for South African languages, incorporating regional accents and tonal nuances

2. **Automated HRTF Processing**: Create dynamic customisation for spatial audio that adapts to individual anatomy and classroom scenarios

3. **Cultural Context Integration**: Incorporate cultural awareness into translation algorithms for authentic communication

4. **Expanded Language Support**: Add more South African languages (isiXhosa, Setswana, Sesotho, etc.)

5. **Classroom Deployment**: Pilot testing in active classroom environments with diverse student populations

---

## 📚 Research Context

This project was developed as part of a COS700 research report at the University of Pretoria, supervised by Dr. Linda Marshall. The research explored:

- **Translanguaging** as a pedagogical strategy for multilingual classrooms
- **Perspective Sharing** as a framework for immersive collaborative learning
- **Spatial Audio (HRTF)** for enhanced auditory immersion
- **Smart Learning Environments** as technology-enhanced adaptive learning systems

---

## 📖 References

Key sources informing this research:

- Hevner, A. & Chatterjee, S. *Design Research in Information Systems: Theory and Practice*
- García, O. & Wei, L. *Translanguaging*
- Hwang, G.-J. "Definition, framework and research issues of smart learning environments"
- Pieterse, T. & Marshall, L. "Modelling Interactive 3d Perspective Sharing for Online Learning"
- Dhawan, S. "Speech To Speech Translation: Challenges and Future"
- Li, S. & Peissig, J. "Measurement of Head-Related Transfer Functions: A Review"

---

## 👥 Authors

- **Joy Azile Zoe Bengu** - *Design, Development & Research* - [@jazbengu](https://github.com/jazbengu)
- **Dr. Linda Marshall** - *Supervisor* - University of Pretoria

---

## 🙏 Acknowledgments

- University of Pretoria, Department of Computer Science
- Dr. Linda Marshall for supervision and guidance
- All testing participants who provided valuable feedback

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Research Report**: [COS700_Research_Report_2500307.pdf](COS700_Research_Report_2500307.pdf)
- **GitHub Repository**: [https://github.com/jazbengu/ClassifySpeech](https://github.com/jazbengu/ClassifySpeech)

---

*Built with ❤️ for inclusive multilingual education*
