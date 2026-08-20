from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Peano Curve: Construction through Iteration", 
                          ["We define the Peano curve iteratively.", 
                           "Divide the square into nine parts.", 
                           "Connect centers recursively each step.", 
                           "Complexity grows with each iteration.", 
                           "The curve approaches a dense web."])
        
        # === Animation for Lecture Line 1 ===
        # Use asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg as requested
        line_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg", color=WHITE)
        self.place_in_area(line_icon, 'B3', 'B4', scale_factor=0.7)
        self.play(Create(line_icon))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        squares = VGroup(*[Square(side_length=0.4, color="#32CD32") for _ in range(9)])
        squares.arrange_in_grid(3, 3, buff=0.05)
        self.place_in_area(squares, 'C2', 'E4', scale_factor=0.5)
        self.play(FadeIn(squares))
        self.lecture[1].set_color("#32CD32")

        # === Animation for Lecture Line 3 ===
        path = VMobject(color="#FFD700")
        path.set_points_smoothly([squares[i].get_center() for i in [0, 1, 2, 5, 4, 3, 6, 7, 8]])
        self.play(Create(path))
        self.lecture[2].set_color("#FFD700")

        # === Animation for Lecture Line 4 ===
        path2 = path.copy().scale(0.3).next_to(path, RIGHT, buff=0.1)
        self.play(TransformFromCopy(path, path2))
        self.lecture[3].set_color("#FFD700")

        # === Animation for Lecture Line 5 ===
        # Use asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg as requested
        web = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg", color="#FF4500")
        self.place_in_area(web, 'C2', 'F5', scale_factor=0.6)
        self.play(FadeIn(web))
        self.lecture[4].set_color("#FF4500")
