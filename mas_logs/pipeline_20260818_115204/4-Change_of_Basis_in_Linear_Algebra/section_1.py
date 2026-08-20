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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Vectors stay the same, but grids change.", 
                         "Robot at point P, standard basis (3,2).", 
                         "Robot at point P, rotated basis (2,1)."]
        self.setup_layout("Intuitive Hook: The Perspective Shift", lecture_lines)
        
        # Assets
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        
        # Define objects
        grid = NumberPlane(x_range=[-4, 4], y_range=[-4, 4], background_line_style={"stroke_opacity": 0.3})
        self.place_in_area(grid, 'B3', 'F6', scale_factor=0.5)
        self.add(grid)
        
        vector = Vector([2, 1], color=WHITE)
        self.place_at_grid(vector, "D3")
        label_v = MathTex("V", color=WHITE).next_to(vector.get_end(), UP)
        
        point_P = Dot(color=YELLOW)
        self.place_at_grid(point_P, 'C2', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        # Load and place robot
        robot_start = robot.copy()
        self.place_at_grid(robot_start, 'C3', scale_factor=0.3)
        self.play(Create(vector), Write(label_v), FadeIn(robot_start))
        self.play(Indicate(vector, color="#FFFF00"))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.play(FadeIn(point_P)) 
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        new_grid = NumberPlane(x_range=[-4, 4], y_range=[-4, 4], background_line_style={"stroke_opacity": 0.3})
        new_grid.rotate(PI/4)
        self.place_in_area(new_grid, 'A4', 'F6', scale_factor=0.5)
        
        robot_end = robot.copy()
        self.place_at_grid(robot_end, 'E4', scale_factor=0.3)
        
        self.play(Transform(grid, new_grid), FadeIn(robot_end))
        label_v_prime = MathTex("V'", color="#00FFFF").next_to(vector.get_end(), UP)
        self.play(ReplacementTransform(label_v, label_v_prime))
        self.wait(2)
