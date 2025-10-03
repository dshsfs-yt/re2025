from typing import List
import copy

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
                '''if (self.L, j) in L_DOUBLE_FROM:
                    self.L = L_DOUBLE_FROM[(self.L, j)]
                    self.stay = self.stay[:-1]
                    self.stay.append(self.L)
                    return'''
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
        '''# 마지막이 완성형 음절이면 분해해서 버퍼로 옮긴 뒤, 자모 하나 제거
        if '가' <= last <= '힣':
            L, V, T = decompose_syllable(last)
            self.L, self.V, self.T = L, V, T
            # 한번 더 backspace 로직 수행(자모 하나 제거)
            self.backspace()
            # 조합 중인 상태 유지 (사용자가 이어서 다시 칠 수 있게)
        else:
            # 호환자모나 공백 등의 단일 토큰은 그냥 삭제로 종료
            return
'''

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
def automata(labels: str) -> str:
    """
    labels: ['ㅇ','ㄴ','ㅏ','ㄴ','ㅠ','ㅠ'] 처럼 자모/스페이스/백스페이스 순서
            스페이스는 '[SPACE]', 백스페이스는 '[BKSP]' 문자열 사용
    """
    labels = split_token_stream(labels)
    A = HangulAutomaton()
    i = 0
    for lab in labels:
        if lab == '[SPACE]':
            A.input_space(i)
        elif lab == '[BKSP]':
            #print("# ----------------------------------")
            #print(f"{A.get_text()}\n{A.get_logs()}, {A.stay}")
            A.backspace()
            #print(f"{A.get_text()}\n{A.get_logs()}, {A.stay}")
            #print("# ----------------------------------")
        elif lab != "[MISS]":
            A.input_char(lab, i)
        '''if A.result:
            print(A.get_logs()[-1], A.stay, i)'''
        i += 1
    return A.get_text()


def split_token_stream(s: str) -> List[str]:
    """연속 문자열에서 [SPACE] 같은 대괄호 토큰을 하나로 취급하여 리스트로 분할"""
    out = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "[":
            j = s.find("]", i+1)
            if j != -1:
                out.append(s[i:j+1])
                i = j + 1
                continue
        out.append(s[i])
        i += 1
    return out

