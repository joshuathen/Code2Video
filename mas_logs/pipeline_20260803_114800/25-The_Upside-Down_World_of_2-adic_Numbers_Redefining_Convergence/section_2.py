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
        self.setup_layout(
            "Prerequisite: The Power of Divisibility", 
            [
                "The 2-adic valuation counts powers of two.",
                'Numbers divisible by high powers of two are "small".',
                "A large exponent makes a number virtually weightless."
            ]
        )
        
        # Colors for highlighting
        COLOR_LINE_1 = "#90EE90"  # Light green
        COLOR_LINE_2 = "#ADD8E6"  # Light blue
        COLOR_LINE_3 = "#FFB6C1"  # Light pink
        COLOR_ASSETS = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_LINE_1))
        
        # Formula: v_2(n) = Exponent of 2
        # Fixed positioning per Issue 25: Move to central row C to avoid top-heavy layout.
        formula = MathTex(r"v_2(n) = \text{Exponent of } 2", color=COLOR_LINE_1)
        self.place_in_area(formula, 'C2', 'C5', scale_factor=0.8)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_LINE_2)
        )
        
        # Create a horizontal scale (a simple line)
        scale_line = Line(self.grid["D1"], self.grid["D6"], color=GREY)
        label_2 = MathTex("2", color=COLOR_ASSETS)
        label_4 = MathTex("4", color=COLOR_ASSETS)
        label_1024 = MathTex("1024", color=COLOR_ASSETS)
        
        self.place_at_grid(label_2, "D1", scale_factor=1.0)
        self.place_at_grid(label_4, "D2", scale_factor=1.0)
        # Fixed positioning per Issue 26: Move label_1024 to D4 to reduce the visual gap.
        self.place_at_grid(label_1024, 'D4', scale_factor=1.0)
        
        self.play(Create(scale_line))
        self.play(
            FadeIn(label_2),
            FadeIn(label_4),
            FadeIn(label_1024)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_LINE_3)
        )
        
        # Scale '2' to be large and '1024' to be tiny
        # Storyboard specifies #FFFFFF for these scaled labels, which they already are.
        self.play(
            label_2.animate.scale(1.5),
            label_4.animate.scale(0.8),
            label_1024.animate.scale(0.3)
        )
        
        self.wait(2)
