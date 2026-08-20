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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Mapping Moves to the Sierpinski Triangle", [
            "Map disk configurations to points on a triangle graph.", 
            "Legal moves represent edges between adjacent sub-triangles.", 
            "The entire state space forms a Sierpinski triangle."
        ])
        
        # Define triangle, scaled as per Critic
        triangle = Triangle(color=WHITE)
        self.place_in_area(triangle, 'B3', 'E6', scale_factor=1.2)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(triangle))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg]
        disk_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")
        disk_icons = VGroup(*[disk_icon.copy().scale(0.3).move_to(triangle.get_vertices()[i]) for i in range(3)])
        
        self.play(FadeIn(disk_icons))
        self.play(self.lecture[1].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 3 ===
        sub_triangles = VGroup(\
            Triangle(color=BLUE).scale(0.5).move_to(triangle.get_center() + UP*0.3),\
            Triangle(color=BLUE).scale(0.5).move_to(triangle.get_center() + LEFT*0.3 + DOWN*0.2),\
            Triangle(color=BLUE).scale(0.5).move_to(triangle.get_center() + RIGHT*0.3 + DOWN*0.2)\
        )
        self.play(Create(sub_triangles))
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.wait(2)
