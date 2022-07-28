# https://wikidocs.net/165950
# lambda form with map(), reduce(), filter()


def read(text):
    ridename, limit = list(map(str.strip, text.split(":")))
    cmin = cmax = None

    if '~' in limit:
        cmin, cmax = map(lambda x : int(x.replace('cm','')), limit.split('~'))
    elif '이상' in limit:
        cmin = int(limit.split('cm')[0])
    elif '이하' in limit:
        cmax = int(limit.split('cm')[0])
    
    return ridename, cmin, cmax

if __name__ == "__main__":
    ridename, cmin, cmax = read(input())
    print("이름: ", ridename)
    print("하한: ", cmin)
    print("상한: ", cmax )

