foo = 3
def hey():
    a = 3
    def yo():
        global foo
        global a
        foo += a
    yo()
    global foo 
    foo +=1 

hey()
print(foo)
