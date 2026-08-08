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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Powers, roots, and logs are unified.",
            "Visualize b, x, and y together.",
            "Rearranging creates different perspectives.",
            "One relationship, three different forms.",
            "This is the power triangle."
        ]
        self.setup_layout("The Unified Triangle", lecture_lines)
        
        # Triangle elements
        triangle = Polygon(self.grid["B3"], self.grid["E2"], self.grid["E4"], color=WHITE)
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        triangle_group = VGroup(triangle, prism)
        self.place_in_area(triangle_group, 'C3', 'F5', scale_factor=0.6)
        
        base_label = MathTex("b", color=WHITE)
        exp_label = MathTex("x", color="#FF0000")
        total_label = MathTex("y", color="#00FF00")
        labels = VGroup(base_label, exp_label, total_label)
        self.place_at_grid(labels, 'B3', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW), FadeIn(triangle_group))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE), Write(labels))
        
        # === Animation for Lecture Line 3 ===
        arrow1 = CurvedArrow(triangle.get_vertices()[0], triangle.get_vertices()[1], angle=-TAU/6, color=WHITE)
        arrow2 = CurvedArrow(triangle.get_vertices()[1], triangle.get_vertices()[2], angle=-TAU/6, color=WHITE)
        arrow3 = CurvedArrow(triangle.get_vertices()[2], triangle.get_vertices()[0], angle=-TAU/6, color=WHITE)
        self.play(self.lecture[2].animate.set_color(GREEN), Create(arrow1), Create(arrow2), Create(arrow3))
        
        # === Animation for Lecture Line 4 ===
        equations = VGroup(
            MathTex("b^x = y"),
            MathTex("y^{1/x} = b"),
            MathTex("\\log_b(y) = x")
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.place_in_area(equations, 'C5', 'F6', scale_factor=0.75)
        self.play(self.lecture[3].animate.set_color(ORANGE), Write(equations))
        
        # === Animation for Lecture Line 5 ===
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        compass.scale(0.5).next_to(triangle_group, UP)
        self.play(self.lecture[4].animate.set_color(PURPLE), Indicate(triangle_group), FadeIn(compass))
        self.play(Rotating(triangle_group, radians=TAU/8, about_point=triangle_group.get_center()))
        self.wait(2)
