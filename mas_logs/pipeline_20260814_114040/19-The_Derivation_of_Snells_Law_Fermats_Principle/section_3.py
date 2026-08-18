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
        self.setup_layout("Setting the Geometric Model", [
            "Define two media with different speeds.",
            "Point A to B via boundary.",
            "Define incident and refracted angles."
        ])
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        icon_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color=YELLOW)
        icon_b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color=YELLOW)
        
        # Refined Geometric Model Elements
        boundary = Line(LEFT*2.0, RIGHT*2.0, color=WHITE)
        label1 = Text("n1", font_size=20, color=BLUE).next_to(boundary, UP)
        label2 = Text("n2", font_size=20, color=RED).next_to(boundary, DOWN)
        
        geometric_model = VGroup(boundary, label1, label2)
        self.place_in_area(geometric_model, 'B4', 'E6', scale_factor=1.0)
        
        point_A = self.place_at_grid(icon_a, 'A4', scale_factor=0.8)
        point_B = self.place_at_grid(icon_b, 'F6', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(geometric_model))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        refraction_point = self.grid['C5'] # Boundary on the grid
        path_a = Line(point_A.get_center(), refraction_point, color=WHITE)
        path_b = Line(refraction_point, point_B.get_center(), color=WHITE)
        self.play(Create(path_a), Create(path_b))
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        normal = DashedLine(UP*1.5, DOWN*1.5, color=GREY).move_to(refraction_point)
        angle1 = Arc(radius=0.4, start_angle=PI/2, angle=-PI/3, color=BLUE).move_to(refraction_point)
        angle2 = Arc(radius=0.4, start_angle=-PI/2, angle=PI/4, color=RED).move_to(refraction_point)
        
        theta1 = MathTex(r"\\theta_1", font_size=20, color=BLUE)
        theta2 = MathTex(r"\\theta_2", font_size=20, color=RED)
        
        self.place_at_grid(theta1, 'B3', scale_factor=0.6)
        self.place_at_grid(theta2, 'D3', scale_factor=0.6)
        
        self.play(Create(normal), Create(angle1), Create(angle2), Write(theta1), Write(theta2))
        self.lecture[2].set_color(RED)
        self.wait(2)
