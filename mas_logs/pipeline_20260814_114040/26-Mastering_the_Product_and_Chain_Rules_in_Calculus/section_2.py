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
        lecture_lines = [
            "The Product Rule finds the derivative of f(x)g(x).",
            "It equals f'(x)g(x) plus f(x)g'(x).",
            "Visualize changes as adding rectangle strips."
        ]
        self.setup_layout("The Product Rule", lecture_lines)
        
        # Colors
        COLOR_BLUE = "#3498DB"
        COLOR_GREEN = "#2ECC71"
        COLOR_WHITE = "#FFFFFF"

        # Asset path
        ASSET_RECT = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/rectangle.svg"

        # === Animation for Lecture Line 1 ===
        # Use asset for "product rule formula"
        formula_base = SVGMobject(ASSET_RECT, color=COLOR_WHITE)
        eq_text = MathTex(r"\\frac{d}{dx} [f(x)g(x)]", font_size=36)
        eq1 = VGroup(formula_base, eq_text).arrange(DOWN)
        self.place_in_area(eq1, 'B3', 'B5', scale_factor=0.9)
        self.play(Write(eq1))
        self.lecture[0].set_color(COLOR_WHITE)

        # === Animation for Lecture Line 2 ===
        # Highlights: f'(x)g(x) in blue, f(x)g'(x) in green
        eq2_part1 = SVGMobject(ASSET_RECT, color=COLOR_BLUE)
        eq2_text1 = MathTex(r"f'(x)g(x)", color=COLOR_BLUE, font_size=36)
        eq2_1 = VGroup(eq2_part1, eq2_text1).arrange(DOWN)
        
        eq2_part2 = SVGMobject(ASSET_RECT, color=COLOR_GREEN)
        eq2_text2 = MathTex(r"f(x)g'(x)", color=COLOR_GREEN, font_size=36)
        eq2_2 = VGroup(eq2_part2, eq2_text2).arrange(DOWN)
        
        eq2 = VGroup(eq2_1, MathTex(r"+"), eq2_2).arrange(RIGHT)
        self.place_in_area(eq2, 'C3', 'C5', scale_factor=0.9)
        
        self.play(Write(eq2))
        self.lecture[1].set_color(COLOR_BLUE) # Simplified representation

        # === Animation for Lecture Line 3 ===
        rect1 = SVGMobject(ASSET_RECT, color=COLOR_BLUE, fill_opacity=0.3)
        rect2 = SVGMobject(ASSET_RECT, color=COLOR_GREEN, fill_opacity=0.3)
        
        self.place_at_grid(rect1, 'E3', scale_factor=0.7)
        self.place_at_grid(rect2, 'E5', scale_factor=0.7)
        
        self.play(Create(rect1), Create(rect2))
        self.lecture[2].set_color(COLOR_WHITE)
