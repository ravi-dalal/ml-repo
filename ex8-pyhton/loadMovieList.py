def loadMovieList():

    fid = open('data/movie_ids.txt', 'r')

    movieList = {}
    for line in fid.readlines():
        i, movieName = line.split(" ", 1)
        movieList[int(i)-1] = movieName.strip()

    fid.close()
    
    return movieList