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
        lecture_lines = ["Circles of all sizes share a secret ratio.", "Small hamster wheels, large Ferris wheels alike.", "This special ratio is always the same."]
        self.setup_layout("The Hook: The Constant Ratio", lecture_lines)
        
        # Load Assets
        hamster = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hamster.svg")
        wheel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wheel.svg")
        
        circle = Circle(radius=1.5, color="#FFFFFF")
        circle_group = VGroup(circle, hamster)
        self.place_in_area(circle_group, 'B3', 'E5', scale_factor=0.75)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(circle_group), self.lecture[0].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 2 ===
        # Represent diameter and radius
        radius_line = Line(circle.get_center(), circle.get_right(), color="#FF0000")
        radius_label = Text("r", font_size=20, color="#FF0000").next_to(radius_line, UP)
        diameter_line = Line(circle.get_left(), circle.get_right(), color="#00FF00")
        diameter_label = Text("d", font_size=20, color="#00FF00").next_to(diameter_line, DOWN)
        
        self.play(Create(radius_line), Write(radius_label), self.lecture[1].animate.set_color("#00FF00"))
        self.play(Create(diameter_line), Write(diameter_label))
        
        # === Animation for Lecture Line 3 ===
        d_eq_2r = MathTex("d = 2r", color="#FFFFFF")
        self.place_at_grid(d_eq_2r, 'E4', scale_factor=0.9)
        self.place_at_grid(wheel, 'A3', scale_factor=0.5)
        
        self.play(Write(d_eq_2r), FadeIn(wheel), self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(2)
