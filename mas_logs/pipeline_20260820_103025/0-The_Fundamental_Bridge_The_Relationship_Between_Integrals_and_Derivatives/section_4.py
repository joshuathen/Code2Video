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
        self.setup_layout("The Inverse Operation: Differentiation and Integration", 
                          ["Differentiation and integration are inverse operations.", 
                           "Balloon volume tracks its expansion rate.", 
                           "Antiderivatives reverse the growth process."])
        
        balloon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/balloon.svg"
        
        # === Animation for Lecture Line 1 ===
        F_x = MathTex("F(x)", color="#FFFFFF")
        f_x = MathTex("f(x) = F'(x)", color="#FFFFFF")
        arrow = Arrow(start=LEFT, end=RIGHT, color=WHITE)
        balloon = SVGMobject(balloon_path, color=WHITE)
        
        formula_group = VGroup(F_x, arrow, f_x, balloon).arrange(RIGHT, buff=0.3)
        self.place_in_area(formula_group, 'B2', 'B5', scale_factor=0.9)
        
        self.play(Write(F_x), GrowArrow(arrow), Write(f_x), FadeIn(balloon))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        reverse_arrow = Arrow(start=RIGHT, end=LEFT, color="#FF0000")
        self.place_at_grid(reverse_arrow, "C3")
        
        int_symbol = MathTex(r"\\int f(x) dx", color="#FF0000")
        self.place_at_grid(int_symbol, 'C2', scale_factor=1.0)
        
        self.play(GrowArrow(reverse_arrow), Write(int_symbol))
        self.play(self.lecture[1].animate.set_color("#FF0000"))

        # === Animation for Lecture Line 3 ===
        final_formula = MathTex(r"\\frac{d}{dx} \\int f(x) dx = f(x)", color="#00FF00")
        self.place_in_area(final_formula, 'D2', 'D5', scale_factor=0.85)
        
        self.play(Write(final_formula), FadeOut(balloon))
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.wait(2)
