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
        self.setup_layout("The Concept of Iteration", [
            "Iteration means repeatedly applying the same function.",
            "Orbits trace the path of a starting point.",
            "Simple rules create complex, dynamic sequences."
        ])
        
        # Fixed axes positioning as per VideoCritic
        axes = Axes(x_range=[-2, 2, 1], y_range=[-2, 2, 1], x_length=4, y_length=4)
        self.place_in_area(axes, 'B3', 'E6', scale_factor=0.7)
        self.add(axes)

        # Assets as per instructions
        # Note: Since the prompt uses placeholder /none.svg, we interpret these as markers 
        # or icons to add at the grid positions.

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        z0_dot = Dot(color="#FFFFFF")
        # Fixed z0_dot pos as per VideoCritic
        self.place_at_grid(z0_dot, 'D3', scale_factor=0.6)
        
        z0_label = MathTex("z_0 = 0")
        # Fixed z0_label pos as per VideoCritic
        self.place_at_grid(z0_label, 'D4', scale_factor=0.5)
        
        self.play(Create(z0_dot), Write(z0_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFA500"))
        
        c = complex(-0.5, 0.5)
        z = 0
        points = [z]
        for _ in range(5):
            z = z**2 + c
            points.append(z)
            
        dots = VGroup()
        for p in points:
            dot = Dot(axes.c2p(p.real, p.imag), color="#FFA500", radius=0.06)
            dots.add(dot)
            
        self.play(FadeIn(dots))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        path = VMobject(color="#00FF00", stroke_width=2)
        path.set_points_smoothly([axes.c2p(p.real, p.imag) for p in points])
        self.play(Create(path))
        self.wait(2)
