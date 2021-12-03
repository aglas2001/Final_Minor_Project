$start  = 'C:\Users\adria\Desktop\CSE_Minor_Project\dockerjive\Newfolder'
cd $start
ls


foreach($child in (ls "$start" -Recurse  -Directory))
{
     $target_directory = $start + '\' +  $child
     echo $target_directory
     cd $target_directory
     echo "About to ls"
     ls
    docker run --rm -v ${PWD}:/tmp csefem fem-opt rve.pro
    cd ..
}



$main = 'C:\Users\adria\Desktop\CSE_Minor_Project'
cd $main
