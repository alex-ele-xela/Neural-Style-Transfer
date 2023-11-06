def test(n):
    if n<=0:
        return []
    
    elif n==1:
        return [0]
    
    else:
        fit = [0,1]
        while len(fit) < n:
            fit.append(fit[-1] + fit[-2])