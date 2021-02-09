using Distances, Statistics
using MultivariateStats
using PyPlot
using WordTokenizers
using TextAnalysis
using DelimitedFiles

function load_file(embedding_file)
    local LL, indexed_words, index
    indexed_words = Vector{String}()
    LL = Vector{Vector{Float32}}()
    open(embedding_file) do f
        index = 1
        for line in eachline(f)
            xs = split(line)
            word = xs[1]
            push!(indexed_words, word)
            push!(LL, parse.(Float32, xs[2:end]))
            index += 1
        end
    end
    return reduce(hcat, LL), indexed_words
end

wrd_embed, libb = load_file("C:/Users/Chetan/Desktop/glove.6B.50d.txt")
vec_size, libb_size = size(wrd_embed)
println("Downloading the word vocabulary adn peseenting as  $vec_size features. and its size is = $libb_size")

vec_idx(s) = findfirst(x -> x==s, libb)
vec_idx("chocolate")

function vec(x) 
    if vec_idx(x)!=nothing
        wrd_embed[:, vec_idx(x)]
    end    
end
vec("chocolate")

cos_rever_dis(x,y)=1-cosine_dist(x, y)

cos_rever_dis(vec("wine"), vec("grapes")) > cosine(vec("chocolate"),vec("octopus"))

function near(q, k=5)
    list=[(z,cos_rever_dis(wrd_embed'[z,:], q)) for z in 1:size(wrd_embed)[2]]
    topn_idx=sort(list, by = z -> z[2], rev=true)[1:k]
    return [libb[a] for (a,_) in topn_idx]
end

near(vec("chocolate")) #for 5 similar words to the input

near(vec("tree"))

week_cck = vec("week") - vec("weekend")
near(week_cck + vec("monday"))

queen_check = vec("king") - vec("man")
near(queen_check + vec("woman"))#correct answer is predicted to closest apporoximation ie. 2nd word of the array list creaeted


