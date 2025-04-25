import numpy as np
from roboticstoolbox.robot.Robot import Robot
from spatialmath import SE3
from typing import Union, List
from roboticstoolbox.robot.Link import Link
from numpy.typing import ArrayLike



class Panda_qdecay(Robot):

    def __init__(self):
        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "franka_description/robots/panda_arm_hand.urdf.xacro"
        )


        super().__init__(
            links,
            name=name,
            manufacturer="Franka Emika",
            gripper_links=links[9],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.grippers[0].tool = SE3(0, 0, 0.1034)

        self.qdlim = np.array(
            [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0]
        )

        self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        self.qz = np.zeros(7)


        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

    def link_collision_damper(                   
        shape,                
        q: ArrayLike,          
        di: float = 0.3,       
        ds: float = 0.03,     
        xi: float = 1.0,       
        end: Union[Link, None] = None,   
        start: Union[Link, None] = None, 
        collision_list: Union[List, None] = None, 
    ):
      
        end, start, _ = self._get_limit_links(start=start, end=end)
       
        links, n, _ = self.get_path(start=start, end=end)

       
        q = np.array(q)
        j = 0
        Ain = None
        bin = None
        min_distance = float('inf')  

        def indiv_calculation(link: Link, link_col, q):

            try:
               
                d, wTlp, wTcp = link_col.closest_point(shape, di) 
                if d is None or wTlp is None or wTcp is None:
                    return None, None, float('inf')

                
                lpTcp = -wTlp + wTcp  

              
                norm = lpTcp / d if d > 0 else np.zeros(3) 

                norm_h = np.expand_dims(
                    np.concatenate((norm, [0.0, 0.0, 0.0])),
                    axis=0
                )

 
                Je = self.jacobe(q, start=start, end=link, tool=link_col.T)   
                                                                              
                n_dim = Je.shape[1]  

                
                dp = norm_h @ shape.v  

              
                l_Ain = np.zeros((1, n))
                l_Ain[0, :n_dim] = norm_h @ Je  

            
                l_bin = (xi * (d - ds) / (di - ds)) + dp     

                return l_Ain, l_bin, d
            except Exception as e:
               
                print(f"Error in indiv_calculation: {e}")
                return None, None, float('inf')

        for link in links:

            if link.isjoint:
                j += 1 

            if collision_list is None:
                col_list = link.collision  
            else:
                col_list = [collision_list[j - 1]] 

            for link_col in col_list:
                l_Ain, l_bin, d = indiv_calculation(link, link_col, q)

                if d is not None:
                    min_distance = min(min_distance, d)


                if l_Ain is not None and l_bin is not None:
                    if Ain is None:
                        Ain = l_Ain
                    else:
                        Ain = np.concatenate((Ain, l_Ain))

                    if bin is None:
                        bin = np.array(l_bin)
                    else:
                        bin = np.concatenate((bin, l_bin))


        return Ain, bin, min_distance

if __name__ == "__main__":  

    r = Panda_qdecay()

    r.qz

    for link in r.grippers[0].links:
        print(link)
    

