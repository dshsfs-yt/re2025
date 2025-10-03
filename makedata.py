from __future__ import annotations
import json,unicodedata
from typing import  Dict, List
import copy
from pathlib import Path


INPUT_DIR       = Path("saved_logs")    # JSON들이 있는 폴더
OUTPUT_DIR      = Path("saved_logs_nokk")   # 저장할 폴더
RECURSIVE       = True                 # 하위 폴더까지 처리할지
KEEP_STRUCTURE  = True                 # 입력 폴더 구조를 보존할지
PATTERN         = "*.json"             # 처리할 파일 패턴
INDENT          = 2                    # pretty 출력 (compact: None 또는 -1)
ENCODING        = "utf-8"              # 파일 인코딩

# hangul_automaton.py

# --- 1) 한글 조합 테이블 (호환 자모 기준) ---
CHOSEONG = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
JUNGSEONG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
JONGSEONG = ['', 'ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

# 복모음 분해/결합 (앞자+뒤자 -> 복모음)
V_COMBINE = {
    ('ㅗ','ㅏ'):'ㅘ', ('ㅗ','ㅐ'):'ㅙ', ('ㅗ','ㅣ'):'ㅚ',
    ('ㅜ','ㅓ'):'ㅝ', ('ㅜ','ㅔ'):'ㅞ', ('ㅜ','ㅣ'):'ㅟ',
    ('ㅡ','ㅣ'):'ㅢ'
}
# 복모음 분해: 복모음 -> (앞, 뒤)
V_SPLIT = {v:k for k,v in V_COMBINE.items()}

# 종성 쌍자음 결합/분해
T_COMBINE = {
    ('ㄱ','ㅅ'):'ㄳ',
    ('ㄴ','ㅈ'):'ㄵ', ('ㄴ','ㅎ'):'ㄶ',
    ('ㄹ','ㄱ'):'ㄺ', ('ㄹ','ㅁ'):'ㄻ', ('ㄹ','ㅂ'):'ㄼ', ('ㄹ','ㅅ'):'ㄽ', ('ㄹ','ㅌ'):'ㄾ', ('ㄹ','ㅍ'):'ㄿ', ('ㄹ','ㅎ'):'ㅀ',
    ('ㅂ','ㅅ'):'ㅄ'
}
T_SPLIT = {v:k for k,v in T_COMBINE.items()}

# 초성 쌍자음(된소리) 분해/결합 (입력으로 ㅉ, ㄲ, ㅆ, ㅃ, ㄸ가 직접 들어올 수도 있고, 같은 자음 두 번으로도 취급)
L_DOUBLE_FROM = {('ㄱ','ㄱ'):'ㄲ', ('ㄷ','ㄷ'):'ㄸ', ('ㅂ','ㅂ'):'ㅃ', ('ㅅ','ㅅ'):'ㅆ', ('ㅈ','ㅈ'):'ㅉ'}
L_DOUBLE_SPLIT = {'ㄲ':'ㄱ', 'ㄸ':'ㄷ', 'ㅃ':'ㅂ', 'ㅆ':'ㅅ', 'ㅉ':'ㅈ'}

BASE = 0xAC00

def is_choseong(j): return j in CHOSEONG
def is_jungseong(j): return j in JUNGSEONG
def is_jongseong(j): return j in JONGSEONG[1:]  # 빈종성 제외
def to_L_index(j): return CHOSEONG.index(j)
def to_V_index(j): return JUNGSEONG.index(j)
def to_T_index(j): return JONGSEONG.index(j)

def compose_syllable(L: str, V: str, T: str|None=None) -> str:
    """초성 L, 중성 V, (선택)종성 T로 완성형 음절 생성"""
    l = to_L_index(L)
    v = to_V_index(V)
    t = to_T_index(T) if T else 0
    code = BASE + (l*21 + v)*28 + t
    return chr(code)

def decompose_syllable(s: str) -> tuple[str,str,str|None]:
    """완성형 음절을 (초성, 중성, 종성?)으로 분해"""
    code = ord(s) - BASE
    l = code // (21*28)
    v = (code % (21*28)) // 28
    t = code % 28
    L = CHOSEONG[l]
    V = JUNGSEONG[v]
    T = JONGSEONG[t] if t != 0 else None
    return L, V, T

# --- 2) 오토마타 상태 ---
class HangulAutomaton:
    def __init__(self):
        self.output: list[str] = []  # 이미 확정된 문자들의 리스트
        # 현재 조합 중인 버퍼(자모 단위로 기억)
        self.L: str|None = None
        self.V: str|None = None
        self.T: str|None = None

        self.stay = []
        self.result = []


    # ---- 내부 유틸 ----
    def _flush(self):
        """현재 버퍼를 확정(가능하면 완성형), 없으면 무시"""
        if self.L and self.V:
            self.output.append(compose_syllable(self.L, self.V, self.T))
        elif self.L and not self.V:
            # 초성만 존재하면 호환 자모로 확정
            self.output.append(self.L)
        elif self.V and not self.L:
            self.output.append(self.V)
        elif self.T and not (self.L or self.V):
            self.output.append(self.T)
        # 버퍼 초기화
        self.L = self.V = self.T = None

        if self.stay:
            self.result.append(copy.copy(self.stay))
        self.stay = []

    def _commit_char(self, ch: str, i):
        """버퍼 비우고 글자 하나 확정"""
        self._flush()
        self.output.append(ch)

        self.result.append([i])

    # ---- 입력 처리 ----
    def input_space(self, i):
        self._flush()
        self.output.append(' ')

        self.result.append([i])

    def input_char(self, j: str, i: int):
        # 1) 모음 입력
        if is_jungseong(j):
            if self.L and self.V is None:
                # 초성만 있는 상태 -> 중성 채우기
                self.V = j

                self.stay.append(i)
                return
            if self.L and self.V:
                # 이미 중성이 있고 종성 후보가 없다면 복모음 시도
                if self.T is None and (self.V, j) in V_COMBINE:
                    self.V = V_COMBINE[(self.V, j)]


                    self.stay.append(i)

                    return
                # 종성이 있었다면, 종성을 다음 음절의 초성으로 넘기고 현재 음절 확정
                if self.T:
                    # 종성이 쌍자음이면 뒤 자모를 넘김
                    if self.T in T_SPLIT:
                        t1, t2 = T_SPLIT[self.T]
                        # 현재 음절은 t1만 종성으로 확정
                        self.output.append(compose_syllable(self.L, self.V, t1))

                        # 다음 음절 버퍼 시작: 초성 = t2, 중성 = 현재 입력 모음
                        self.L, self.V, self.T = t2, j, None

                        stay_1, stay_2 = self.stay[:-1], self.stay[-1]
                        self.result.append(copy.copy(stay_1))
                        self.stay = [stay_2, i]
                        return
                    else:
                        # 단일 종성 전체를 초성으로 넘김
                        carry = self.T
                        self.output.append(compose_syllable(self.L, self.V, None))
                        self.L, self.V, self.T = carry, j, None

                        stay_1, stay_2 = self.stay[:-1], self.stay[-1]
                        self.result.append(copy.copy(stay_1))
                        self.stay = [stay_2, i]

                        return
                # 종성 없고 복모음도 못 만들면: 현재 음절 확정 후 새 음절(모음만) 시작
                self.output.append(compose_syllable(self.L, self.V))
                self.L, self.V, self.T = None, j, None

                self.result.append(copy.copy(self.stay))
                self.stay = [i]
                return
            if (self.L is None) and (self.V is None) and (self.T is None):
                # 아무것도 없으면 모음 단독 시작 (스마트폰 자판에서 가능)
                self.V = j

                self.stay = [i]
                return
            # 그 외엔 모두 확정하고 새로 시작
            self._flush()
            self.V = j

            self.stay = [i]
            return

        # 2) 자음 입력
        if is_choseong(j) or is_jongseong(j):
            # 조합 중인 게 없으면 초성으로 시작
            if self.L is None and self.V is None and self.T is None:
                self.L = j if is_choseong(j) else j  # 호환: 그대로 보관
                # 단, 만약 이전 글자와 같은 자음이 연속되어 초성 쌍자음 형성 가능하면 처리는 아래 분기에서 함

                self.stay = [i]
                return

            # (L만 있는 상태) -> 같은 자음으로 된소리 시도
            if self.L and self.V is None:
                # 아니면 기존 L 확정(자음 단독) 후 새 초성 시작
                self.output.append(self.L)
                self.L = j

                self.result.append(copy.copy(self.stay))
                self.stay = [i]
                return

            # (L,V 상태) -> 종성 채우기 또는 다음 글자 시작
            if self.L and self.V and self.T is None:
                # 종성으로 들어갈 수 있는 자음이면 종성 설정
                if is_jongseong(j):
                    self.T = j
                    self.stay.append(i)
                    return
                # 초성 전용 자모(ㄸ,ㅃ,ㅉ 등) 같은 경우 다음 음절로 분기
                self.output.append(compose_syllable(self.L, self.V))
                self.L, self.V, self.T = j, None, None

                self.result.append(copy.copy(self.stay))
                self.stay = [i]
                return

            # (L,V,T 상태) -> 종성 쌍자음 시도 또는 다음 글자 시작
            if self.L and self.V and self.T:
                # 종성 쌍자음 결합
                key = (self.T, j)
                if key in T_COMBINE:
                    self.T = T_COMBINE[key]

                    self.stay.append(i)
                    return
                # 결합 불가면 현재 음절 확정 후 다음 초성으로 시작
                self.output.append(compose_syllable(self.L, self.V, self.T))
                self.L, self.V, self.T = j, None, None

                self.result.append(copy.copy(self.stay))
                self.stay = [i]
                return

            # (모음만 있는 상태)에서 자음이 오면: 모음 확정 후 새 초성 시작
            if self.V and self.L is None:
                self.output.append(self.V)
                self.L, self.V, self.T = j, None, None

                self.result.append(copy.copy(self.stay))
                self.stay = [i]
                return

            # 기타 케이스: 전부 확정하고 새로 시작
            self._flush()
            self.L = j

            self.stay = [i]

            return

        # 3) 그 밖(예: 이모지 등)은 바로 확정
        self._commit_char(j, i)

    def backspace(self):
        # 1) 조합 중이면 자모 하나만 단계적으로 해제
        if self.L or self.V or self.T:
            #print(self.L, self.V, self.T, self.stay)
            self.stay.pop()
            # 종성부터 되돌리기
            if self.T:
                # 쌍종성이면 뒤 자모만 풀기
                if self.T in T_SPLIT:
                    t1, t2 = T_SPLIT[self.T]
                    self.T = t1
                else:
                    self.T = None
                return
            # 복모음이면 뒤 모음만 풀기
            if self.V and self.V in V_SPLIT:
                v1, v2 = V_SPLIT[self.V]
                self.V = v1  # 뒤 모음은 제거
                return
            # 중성 제거
            if self.V:
                self.V = None
                return
            # 초성 제거
            if self.L:
                self.L = None
                return
            return

        # 2) 조합 중이 아니면 마지막 확정 글자에서 자모 하나 되돌리기
        if not self.output:
            return
        last = self.output.pop()
        self.result.pop()
        

    def get_text(self) -> str:
        # 화면 표시용: 현재 조합중 음절까지 포함해서 리턴
        temp = ''.join(self.output)
        if (self.L or self.V or self.T):
            self.result.append(self.stay)
            if self.L and self.V:
                temp += compose_syllable(self.L, self.V, self.T)
            elif self.L:
                temp += self.L
            elif self.V:
                temp += self.V
            elif self.T:
                temp += self.T
        return temp

    def get_logs(self):
        return self.result

# --- 3) 외부 사용 함수 ---
def run_labels(labels: list[str]) -> str:
    """
    labels: ['ㅇ','ㄴ','ㅏ','ㄴ','ㅠ','ㅠ'] 처럼 자모/스페이스/백스페이스 순서
            스페이스는 '[SPACE]', 백스페이스는 '[BKSP]' 문자열 사용
    """
    A = HangulAutomaton()
    i = 0
    for lab in labels:
        if lab == '[SPACE]':
            A.input_space(i)
        elif lab == '[BKSP]':
            A.backspace()
        elif lab != "[MISS]":
            A.input_char(lab, i)
        i += 1
    return A.get_text() , A.get_logs()



# 완성형 분해용 테이블 (호환 자모로 매핑)
CHOSEONG = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
JUNGSEONG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
JONGSEONG = ['', 'ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

# 호환 자모(Compatibility Jamo) 분류용 집합
COMPAT_VOWELS = set('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')
COMPAT_CONSONANTS = set('ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')

VOWEL_DECOMP: Dict[str, List[str]] = {
    'ㅘ': ['ㅗ', 'ㅏ'],
    'ㅙ': ['ㅗ', 'ㅐ'],
    'ㅚ': ['ㅗ', 'ㅣ'],
    'ㅝ': ['ㅜ', 'ㅓ'],
    'ㅞ': ['ㅜ', 'ㅔ'],
    'ㅟ': ['ㅜ', 'ㅣ'],
    'ㅢ': ['ㅡ', 'ㅣ'],
}
FINAL_DECOMP: Dict[str, List[str]] = {
    'ㄳ': ['ㄱ', 'ㅅ'],
    'ㄵ': ['ㄴ', 'ㅈ'],
    'ㄶ': ['ㄴ', 'ㅎ'],
    'ㄺ': ['ㄹ', 'ㄱ'],
    'ㄻ': ['ㄹ', 'ㅁ'],
    'ㄼ': ['ㄹ', 'ㅂ'],
    'ㄽ': ['ㄹ', 'ㅅ'],
    'ㄾ': ['ㄹ', 'ㅌ'],
    'ㄿ': ['ㄹ', 'ㅍ'],
    'ㅀ': ['ㄹ', 'ㅎ'],
    'ㅄ': ['ㅂ', 'ㅅ'],
}

def decompose_hangul_syllable(ch: str):
    """완성형 한글(가-힣)을 (초성, 중성, 종성[없으면 ''])으로 분해. 아니면 None."""
    cp = ord(ch)
    SBase, LBase, VBase, TBase = 0xAC00, 0x1100, 0x1161, 0x11A7
    LCount, VCount, TCount = 19, 21, 28
    NCount = VCount * TCount  # 588
    SCount = LCount * NCount  # 11172

    s_index = cp - SBase
    if 0 <= s_index < SCount:
        l_index = s_index // NCount
        v_index = (s_index % NCount) // TCount
        t_index = s_index % TCount
        cho = CHOSEONG[l_index]
        jung = JUNGSEONG[v_index]
        jong = JONGSEONG[t_index]
        return cho, jung, jong
    return None

def _expand_vowel(v: str) -> List[str]:
    """합자 모음이면 기본 모음들로 분해, 아니면 그대로."""
    return VOWEL_DECOMP.get(v, [v])

def _expand_final_consonant(t: str) -> List[str]:
    """합자 종성이면 두 자음으로 분해, 아니면 그대로(빈 문자열 포함)."""
    if not t:
        return []
    return FINAL_DECOMP.get(t, [t])

def split_korean(text: str):
    """
    스페이스, 자음, 모음을 분리.
    반환:
      sequence: 원문 순서 기준 자모 열거 (공백류는 "[SPACE]" 토큰)
    - 완성형: 초성 + (중성 합자 완전 분해) + (종성 합자까지 분해)
    - 단일 입력 자모도 합자면 동일 규칙으로 분해
    """
    sequence: List[Dict[str, object]] = []

    for i, ch in enumerate(text):
        # 공백류
        if ch.isspace():
            sequence.append("[SPACE]")
            continue

        # 완성형 한글 → 초/중/종 분해 후, 합자 분해 적용
        dec = decompose_hangul_syllable(ch)
        if dec:
            cho, jung, jong = dec
            sequence.append(cho)
            for v in _expand_vowel(jung):
                sequence.append(v)
            for t in _expand_final_consonant(jong):
                sequence.append(t)
            continue

        # 완성형이 아닌 단일 자모가 들어온 경우에도 합자 분해 적용
        # (예: 'ㅘ' → 'ㅗ','ㅏ',  'ㄳ' → 'ㄱ','ㅅ')
        if ch in VOWEL_DECOMP:
            sequence.extend(VOWEL_DECOMP[ch])
            continue
        if ch in FINAL_DECOMP:
            sequence.extend(FINAL_DECOMP[ch])
            continue

        # 그 외 문자(라틴, 숫자, 기호, 확장 자모 등)는 그대로
        sequence.append(ch)

    return sequence



# 유니코드 블록 범위
_JAMO_BLOCKS = (
    (0x1100, 0x11FF),  # Hangul Jamo (초성/중성/종성 글자)
    (0x3130, 0x318F),  # Hangul Compatibility Jamo (호환 자모)
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
)

def is_hangul_syllable(ch: str) -> bool:
    """완성형 한글 음절 여부 (예: '가', '나', '구')."""
    if not ch:
        return False
    o = ord(ch)
    return 0xAC00 <= o <= 0xD7A3

def is_jamo_char(ch: str) -> bool:
    """자모 블록(현대/호환/확장)에 속하는 1글자인지 여부."""
    if not ch:
        return False
    o = ord(ch)
    return any(a <= o <= b for a, b in _JAMO_BLOCKS)

def is_standalone_jamo_char(ch: str) -> bool:
    """
    '단독 자모' 여부: 완성형 음절이 아니고, 자모 블록에 속하면 True.
    (예: 'ㅋ','ㅠ','ㄱ','ㅏ' → True / '가','지','구' → False)
    """
    return (not is_hangul_syllable(ch)) and is_jamo_char(ch)


def find_standalone_jamo(text: str, normalize: bool = True):
    """ 문자열에서 단독 자모의 인덱스 리스트 반환. normalize=True면 NFC 정규화 후 검사(권장).
    """
    if normalize:
        text = unicodedata.normalize("NFC", text)
    new_sentence = ""
    solos = dict()
    index = 0
    for i, ch in enumerate(text):
        if is_standalone_jamo_char(ch):
            solos[ch] = solos.get(ch, [])
            solos[ch].append(i)
        else:
            new_sentence += ch

        index += len(split_korean(ch))



    return new_sentence, solos


def find_continue_space(text: str):
    before = None
    indexs = []
    sentence = ""
    for i, t in enumerate(text):
        if before == t == " ":
            indexs.append(i)
        else:
            sentence += t
        before = t

    return sentence, indexs




def kk_transform(data):
    sentence = data["target_sentence"]
    logs = []
    i = 0
    for log in data["logs"][0]["logs"]:
        label = log["label"]
        logs.append(label)

        i += 1

    text, index_logs = run_labels(logs)


    new_sentence, jamo_indexs = find_standalone_jamo(sentence)
    jamo_index_plet = [x for sub in jamo_indexs.values() for x in sub]
    new_logs = []


    k = [index_logs[i][0] for i in range(len(index_logs)) if i in jamo_index_plet]
    for index in range(len(data["logs"][0]["logs"])):
        if index not in k:
            log = data["logs"][0]["logs"][index]
            new_logs.append(log)

    result = {
    "target_sentence": new_sentence,
    "completed_count": 1,  # 필요시 원본 값 사용
    "logs": [
        {
            "target": new_sentence,
            "logs": list(new_logs)  # deque였다면 list로 변환
            }
        ]
    }
    return result

def space_transform(data):
    sentence = data["target_sentence"]
    logs = []
    i = 0
    for log in data["logs"][0]["logs"]:
        label = log["label"]
        logs.append(label)

        i += 1

    text, index_logs = run_labels(logs)

    new_sentence, space_indexs = find_continue_space(sentence)
    new_logs = []


    k = [index_logs[i][0] for i in range(len(index_logs)) if i in space_indexs]
    for index in range(len(data["logs"][0]["logs"])):
        if index not in k:
            log = data["logs"][0]["logs"][index]
            new_logs.append(log)

    result = {
    "target_sentence": new_sentence,
    "completed_count": 1,  # 필요시 원본 값 사용
    "logs": [
        {
            "target": new_sentence,
            "logs": list(new_logs)  # deque였다면 list로 변환
            }
        ]
    }
    return result

def enter_transform(data):
    sentence = data["target_sentence"]
    logs = []
    i = 0
    for log in data["logs"][0]["logs"]:
        if log["label"] != "[ENTER]":
            label = log["label"]
            logs.append(log)

            i += 1

    result = {
    "target_sentence": sentence,
    "completed_count": 1,  # 필요시 원본 값 사용
    "logs": [
        {
            "target": sentence,
            "logs": list(logs)  # deque였다면 list로 변환
            }
        ]
    }
    return result




json_files = list(INPUT_DIR.glob("*.json"))

for j in json_files:
    with j.open('r', encoding='utf-8') as f:
        data = json.load(f)

    data = space_transform(data)
    data=enter_transform(data)

    new_data = space_transform( kk_transform(data))

    new_name = f"{j.stem}_2{j.suffix}"
    out_path_new = OUTPUT_DIR / new_name
    out_path_orig = OUTPUT_DIR / j.name
    if data['target_sentence'] == new_data['target_sentence']:
        with out_path_new.open('w', encoding='utf-8') as wf:
            json.dump(data, wf, ensure_ascii=False, indent=2)
    if data["target_sentence"] != new_data["target_sentence"]:
        with out_path_orig.open('w', encoding='utf-8') as wf2:
            json.dump(new_data, wf2, ensure_ascii=False, indent=2)
