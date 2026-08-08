from manim import *
import numpy as np

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
        title_text = "The Mathematical Fragmentations"
        lecture_lines = [
            "Traditional math notation for exponents is fragmented.",
            "We use separate symbols for powers, roots, and logarithms.",
            "Observe how 2, 3, and 8 shift around confusingly.",
            "Superscripts, radicals, and 'log' labels feel disconnected.",
            "What if we could unify these three distinct forms?"
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors for variables (consistent throughout the video)
        # 2: Base (Blue)
        # 3: Exponent (Red)
        # 8: Result (Green)
        COLOR_2 = "#3498DB" 
        COLOR_3 = "#E74C3C" 
        COLOR_8 = "#2ECC71" 
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Use simple MathTex and color by index to avoid LaTeX compilation errors with isolation.
        # This is a robust way to handle symbols like \sqrt and \log that often break with substrings_to_isolate.
        power_formula = MathTex("2^3=8")
        root_formula = MathTex(r"\sqrt[3]{8}=2")
        log_formula = MathTex(r"\log_2(8)=3")
        
        # Manual coloring by index (safest method in Manim Community Edition for complex TeX)
        # Power formula indices: 0:'2', 1:'3', 2:'=', 3:'8'
        p2 = power_formula[0][0].set_color(COLOR_2)
        p3 = power_formula[0][1].set_color(COLOR_3)
        p8 = power_formula[0][3].set_color(COLOR_8)
        
        # Root formula indices: 0:root symbol, 1:'3', 2:'8', 3:'=', 4:'2'
        r3 = root_formula[0][1].set_color(COLOR_3)
        r8 = root_formula[0][2].set_color(COLOR_8)
        r2 = root_formula[0][4].set_color(COLOR_2)
        
        # Log formula indices: 0-2:'log', 3:'2', 4:'(', 5:'8', 6:')', 7:'=', 8:'3'
        l2 = log_formula[0][3].set_color(COLOR_2)
        l8 = log_formula[0][5].set_color(COLOR_8)
        l3 = log_formula[0][8].set_color(COLOR_3)
        
        # Position formulas in a vertical stack on the right side (Column 4)
        self.place_at_grid(power_formula, "B4", scale_factor=1.2)
        self.place_at_grid(root_formula, "C4", scale_factor=1.2)
        self.place_at_grid(log_formula, "D4", scale_factor=1.2)
        
        self.play(FadeIn(power_formula), FadeIn(root_formula), FadeIn(log_formula))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Create arrows connecting the numbers across the three notations to show movement
        # 2: Power (base) -> Root (result) -> Log (base) (Blue Path)
        arrow2a = CurvedArrow(p2.get_left(), r2.get_left(), color=COLOR_2, angle=-PI/4).shift(LEFT*0.2)
        arrow2b = CurvedArrow(r2.get_left(), l2.get_left(), color=COLOR_2, angle=-PI/4).shift(LEFT*0.2)
        
        # 3: Power (exp) -> Root (index) -> Log (result) (Red Path)
        arrow3a = CurvedArrow(p3.get_right(), r3.get_right(), color=COLOR_3, angle=PI/4).shift(RIGHT*0.2)
        arrow3b = CurvedArrow(r3.get_right(), l3.get_right(), color=COLOR_3, angle=PI/4).shift(RIGHT*0.2)
        
        # 8: Power (res) -> Root (arg) -> Log (arg) (Green Path)
        arrow8a = CurvedArrow(p8.get_right(), r8.get_right(), color=COLOR_8, angle=PI/2).shift(RIGHT*0.7)
        arrow8b = CurvedArrow(r8.get_right(), l8.get_right(), color=COLOR_8, angle=PI/2).shift(RIGHT*0.7)
        
        self.play(Create(arrow2a), Create(arrow3a), Create(arrow8a))
        self.play(Create(arrow2b), Create(arrow3b), Create(arrow8b))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Fade out mathematical symbols and arrows, leaving only the core colored numbers
        # This uses list comprehension to identify all submobjects that are NOT the numbers we want to keep.
        numbers = {p2, p3, p8, r3, r8, r2, l2, l8, l3}
        all_objs = [*power_formula[0], *root_formula[0], *log_formula[0]]
        to_fade = VGroup(*[m for m in all_objs if m not in numbers])
        
        self.play(
            FadeOut(to_fade),
            FadeOut(arrow2a), FadeOut(arrow2b),
            FadeOut(arrow3a), FadeOut(arrow3b),
            FadeOut(arrow8a), FadeOut(arrow8b)
        )
        self.wait(2)
