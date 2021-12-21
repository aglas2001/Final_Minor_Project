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



filenameO = "C:/Users/aglas/Local_Documents/GitHub/Final_Minor_Project/DataSet/rve_test/para_1.vtu"

filename = "C:/Users/aglas/Local_Documents/GitHub/Final_Minor_Project/WritingVTU/RveTestPara1Remade.vtu"

ShowParaview(filenameO,'Surface')

ShowParaview(filename,'Surface')






