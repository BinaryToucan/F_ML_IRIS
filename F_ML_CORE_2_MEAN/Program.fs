open System
open Microsoft.ML
open Microsoft.ML.Data

[<CLIMutable>]
type IrisData = {
    [<LoadColumn(0)>] SepLength : float32
    [<LoadColumn(1)>] SepWidth : float32
    [<LoadColumn(2)>] PetLength : float32
    [<LoadColumn(3)>] PetWidth : float32
    [<LoadColumn(4)>] Speris : string
}

/// A type that holds a single model prediction.
[<CLIMutable>]
type IrisPrediction = {
    PredictedLabel : uint32
    Score : float32[]
}
let dataPath = sprintf "%s\\data\\IRIS.csv" Environment.CurrentDirectory

[<EntryPoint>]
let main argv = 

    let mlContext = new MLContext();
    let data = mlContext.Data.LoadFromTextFile<IrisData>(dataPath, separatorChar = ',', hasHeader = true)

    // testing - 0.2
    let partitions = mlContext.Data.TrainTestSplit(data, testFraction = 0.2)

    // set up a learning pipeline
    let pipeline = 
        EstimatorChain()
            .Append(mlContext.Transforms.Concatenate("Features", "SepLength", "SepWidth", "PetLength", "PetWidth"))
            .Append(mlContext.Clustering.Trainers.KMeans(numberOfClusters = 3))

    // train the model on the training data
    let model = partitions.TrainSet |> pipeline.Fit
    
    let metrics = partitions.TestSet |> model.Transform |> mlContext.Clustering.Evaluate

    // show results
    printfn "Параметры модели"
    printfn " AverageDistance:     %f" metrics.AverageDistance
    printfn " DaviesBouldinIndex:  %f" metrics.DaviesBouldinIndex

    let engine = mlContext.Model.CreatePredictionEngine model

    // test dataset
    let flowers = mlContext.Data.CreateEnumerable<IrisData>(partitions.TestSet, reuseRowObject = false) |> Array.ofSeq
    let testFlowers = [ flowers.[1]; flowers.[6]; flowers.[13]; flowers.[16]; flowers.[20]]

    // show predictions
    printfn "Предположение о 5-ти цветках:"
    printfn "  Label\t\t\tPredicted\t                Scores"
    testFlowers |> Seq.iter(fun f -> 
            let p = engine.Predict f
            printf "  %-15s\t%d\t\t" f.Speris p.PredictedLabel
            p.Score |> Seq.iter(fun s -> printf "%f\t" s)
            printfn "")

    0 // return value