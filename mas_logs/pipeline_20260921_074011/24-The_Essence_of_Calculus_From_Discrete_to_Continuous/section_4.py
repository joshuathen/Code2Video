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
        self.setup_layout("The Fundamental Theorem: The Bridge", [
            "Differentiation and integration are inverse processes.",
            "One breaks down, the other adds up.",
            "They form the foundation of calculus."
        ])
        
        # Define objects
        derivative_symbol = MathTex(r"\\frac{d}{dx}f(x)", color="#FFFFFF")
        integral_symbol = MathTex(r"\\int f(x) dx", color="#FFFFFF")
        # Load asset
        bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg", color="#FFFF00")
        
        math_group = VGroup(derivative_symbol, bridge, integral_symbol).arrange(RIGHT, buff=0.5)

        # === Animation for Lecture Line 1 ===
        # Using recommendation 31/32 for balanced layout
        self.place_in_area(math_group, 'C2', 'D5', scale_factor=0.85)
        self.play(FadeIn(derivative_symbol), FadeIn(integral_symbol))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(bridge))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        arrow1 = Arrow(start=derivative_symbol.get_bottom(), end=integral_symbol.get_bottom(), color="#FF0000")
        arrow2 = Arrow(start=integral_symbol.get_top(), end=derivative_symbol.get_top(), color="#FF0000")
        
        self.play(GrowArrow(arrow1), GrowArrow(arrow2))
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        self.wait(2)
