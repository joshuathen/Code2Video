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
            "Ultrametric inequality: distance defies standard triangle intuition.",
            "Triangles are always isosceles with short bases.",
            "You cannot exceed the deepest tunnel's depth."
        ]
        self.setup_layout("The Ultrametric Inequality", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display |x + y|_2 <= max(|x|_2, |y|_2).
        ineq = MathTex(r"|x + y|_2 \leq \max(|x|_2, |y|_2)", font_size=36)
        self.place_in_area(ineq, 'B4', 'B6', scale_factor=0.9)
        self.play(Write(ineq))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        # Load asset and use it to represent triangle vertices
        tunnel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tunnel.svg", color="#00FF00")
        
        dot_origin = tunnel.copy().scale(0.3).move_to(self.grid["E3"])
        dot_x = tunnel.copy().scale(0.3).move_to(self.grid["D2"])
        dot_y = tunnel.copy().scale(0.3).move_to(self.grid["D4"])
        
        line_ox = Line(dot_origin.get_center(), dot_x.get_center(), color="#00FF00")
        line_oy = Line(dot_origin.get_center(), dot_y.get_center(), color="#00FF00")
        line_xy = Line(dot_x.get_center(), dot_y.get_center(), color="#00FF00")
        
        triangle_group = VGroup(line_ox, line_oy, line_xy, dot_origin, dot_x, dot_y)
        
        self.place_in_area(triangle_group, 'C3', 'E5', scale_factor=1.0)
        
        self.play(Create(triangle_group))
        self.play(self.lecture[1].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 3 ===
        # Highlight isosceles nature
        highlight = VGroup(line_ox, line_oy).set_color("#FF00FF")
        self.play(Indicate(highlight))
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.wait(2)
