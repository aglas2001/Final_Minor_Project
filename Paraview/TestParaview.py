import paraview.simple as ps

# files =[]
# for i in range(1,51):
#     string = "C:/Users/aglas/OneDrive/Bureaublad/Documenten/TW Jaar 3/CSE Minor/Final Minor Project/WritingVTU/foo.vtu",

#     files.append(string)

def ShowParaview(filename,repren):
    reader = ps.OpenDataFile(filename)
    
    dp = ps.GetDisplayProperties(reader)
    dp.Representation = repren
    
    # dp.PointData = 'a'
    
    
    ps.Show(reader)
    ps.Render()
    
    #dp.ColorArrayName = ['POINTS', 'Displacement']
    
    scene = ps.GetAnimationScene()
    scene.FramesPerTimestep = 1
    scene.Play()



filename0 = "C:/Users/jobre/OneDrive - Erasmus University Rotterdam/Github_cloned/Final_Minor_Project/DataSet/rve_test/para_1.vtu"

filename1 = "C:/Users/jobre/OneDrive - Erasmus University Rotterdam/Github_cloned/Final_Minor_Project/WritingVTU/RveTestPara1Remade.vtu"

filename2 = "C:/Users/jobre/OneDrive - Erasmus University Rotterdam/Github_cloned/Final_Minor_Project/WritingVTU/RveTestPara1Remade2.vtu"


ShowParaview(filename2,'Surface')

ShowParaview(filename2,'Surface')






