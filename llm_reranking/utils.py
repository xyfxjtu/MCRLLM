from difflib import SequenceMatcher


def restore_and_match(A, C):
    target_len = len(A)
    flag = True
    for c in C:
        if c not in A:
            flag =False
            break
    if len(C)!=target_len:
        flag = False
    if flag:
        return C
    if len(C) < target_len:
        C = C + [None] * (target_len - len(C))
    elif len(C) > target_len:
        C = C[:target_len]
    A_pool = set(A)
    B = []
    used = set()  # 已匹配的A中元素
    for c in C:
        if c is None:
            B.append(None)  # 占位，后续填充
            continue
        # 情况1：完全匹配
        if c in A_pool:
            B.append(c)
            used.add(c)
            A_pool.remove(c)
            continue
        # 情况2：模糊匹配（优先子字符串匹配）
        best_match = None
        max_score = 0
        for a in A_pool:
            # 规则1：c是a的子串 → 最高优先级
            if c in a:
                score = 2.0  # 超高分确保优先
            else:
                score = SequenceMatcher(None, c, a).ratio()
            if score > max_score:
                max_score = score
                best_match = a
        if best_match and max_score >= 0.4:
            B.append(best_match)
            used.add(best_match)
            A_pool.remove(best_match)
        else:
            B.append(None)  # 占位
    remaining = [a for a in A if a not in used]
    for i in range(len(B)):
        if B[i] is None and remaining:
            B[i] = remaining.pop(0)
    for b in B:
        if b not in A:
            return A
    return B