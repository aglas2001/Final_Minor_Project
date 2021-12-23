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
    
    
    display = ps.Show(reader)
    ps.Render()
    
    ps.ColorBy(display, ("Surface", "Displacement"))
    
    scene = ps.GetAnimationScene()
    scene.FramesPerTimestep = 1
    scene.Play()



def ShowComponents(Dimension,location):
    for i in range (Dimension):
        filename = location + "Component_"+str(i+1)+".vtu"
        ShowParaview(filename,'Surface')

        
        
        
        

location = "C:/Users/aglas/Local_Documents/GitHub/Final_Minor_Project/ICA/VTUFiles/Components/"
ShowComponents(4,location)






