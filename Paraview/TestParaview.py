import paraview.simple as ps

files =[]
for i in range(1,51):
    string = "C:/Users/aglas/Local_Documents/GitHub/Final_Minor_Project/DataSet/rve_test/para_"+str(i)+ ".vtu"
    files.append(string)

reader = ps.OpenDataFile(files)



#reader = OpenDataFile("C:/Users/aglas/Local_Documents/GitHub/Final_Minor_Project/DataSet/rve_test/para_1.vtu")

dp = ps.GetDisplayProperties(reader)
dp.Representation = 'Surface'


ps.Show(reader)
ps.Render()

#dp.ColorArrayName = ['POINTS', 'Displacement']

scene = ps.GetAnimationScene()
scene.FramesPerTimestep = 1
scene.Play()



