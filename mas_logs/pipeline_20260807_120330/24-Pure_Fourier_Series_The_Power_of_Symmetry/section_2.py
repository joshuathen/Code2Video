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
        # Initializing the layout
        self.setup_layout(
            "Prerequisite: The Full Fourier Recipe",
            [
                "Every periodic function is a mathematical \"stew\" of waves.",
                "The recipe includes constants, cosines, and sine components.",
                "These building blocks are independent and do not mix."
            ]
        )
        
        # Colors
        YELLOW_A = "#FFFF00"
        BLUE_B = "#0000FF"
        GREEN_C = "#00FF00"
        WHITE_CLR = "#FFFFFF"
        GRAY_CLR = "#888888"

        # === Animation for Lecture Line 1 ===
        # Every periodic function is a mathematical "stew" of waves.
        self.lecture[0].set_color(WHITE_CLR)
        self.lecture[1].set_color(GRAY_CLR)
        self.lecture[2].set_color(GRAY_CLR)
        
        # Full Fourier formula
        # f(x) = a0 + sum ( an cos(nx) + bn sin(nx) )
        formula = MathTex(
            "f(x)", "=", "a_0", "+", "\\sum_{n=1}^{\\infty}", 
            "(", "a_n \\cos(nx)", "+", "b_n \\sin(nx)", ")"
        )
        # Fix for Issue 34: formula too wide, move to B2-C5 and scale down
        self.place_in_area(formula, 'B2', 'C5', scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The recipe includes constants, cosines, and sine components.
        self.play(
            self.lecture[0].animate.set_color(GRAY_CLR),
            self.lecture[1].animate.set_color(WHITE_CLR)
        )
        
        # Highlight a0 in yellow, Cosine terms in blue, Sine terms in green
        self.play(
            formula[2].animate.set_color(YELLOW_A),
            formula[6].animate.set_color(BLUE_B),
            formula[8].animate.set_color(GREEN_C),
            run_time=1.5
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # These building blocks are independent and do not mix.
        self.play(
            self.lecture[1].animate.set_color(GRAY_CLR),
            self.lecture[2].animate.set_color(WHITE_CLR)
        )
        
        # Visual representation of orthogonality
        # Fix for Issue 36: scale_factor adjusted to 0.8
        cos_block = VGroup(
            RoundedRectangle(corner_radius=0.1, height=0.8, width=1.5, color=BLUE_B),
            MathTex("\\cos(nx)", color=BLUE_B).scale(0.7)
        )
        sin_block = VGroup(
            RoundedRectangle(corner_radius=0.1, height=0.8, width=1.5, color=GREEN_C),
            MathTex("\\sin(nx)", color=GREEN_C).scale(0.7)
        )
        
        self.place_at_grid(cos_block, 'E2', scale_factor=0.8)
        self.place_at_grid(sin_block, 'E5', scale_factor=0.8)
        
        # Fix for Issue 35: indep_label placement and centering
        indep_label = Text("Independent", font_size=20, color=WHITE_CLR)
        self.place_in_area(indep_label, 'D3', 'D4', scale_factor=1.0)

        # Transition from formula to blocks
        # Shrink formula and move it up to make room
        # We use a temporary center for the target location to strictly follow positioning rules
        target_center = (self.grid['B3'] + self.grid['B4']) / 2
        
        self.play(
            formula.animate.scale(0.7).move_to(target_center),
            FadeIn(cos_block),
            FadeIn(sin_block),
            Write(indep_label)
        )
        
        # Show they don't mix by pulsing them separately
        self.play(cos_block.animate.scale(1.1), run_time=0.5)
        self.play(cos_block.animate.scale(1/1.1), run_time=0.5)
        self.play(sin_block.animate.scale(1.1), run_time=0.5)
        self.play(sin_block.animate.scale(1/1.1), run_time=0.5)
        
        self.wait(3)
