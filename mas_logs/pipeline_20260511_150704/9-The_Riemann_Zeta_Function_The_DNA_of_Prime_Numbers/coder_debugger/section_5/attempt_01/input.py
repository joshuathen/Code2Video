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

class Section5Scene(TeachingScene):
    def construct(self):
        # Section Context
        title = "The Million Dollar Mystery: The Riemann Hypothesis"
        lines = [
            "Most zeros sit on a single vertical line.",
            "This critical line is where the real part is half.",
            "Riemann hypothesized that every non-trivial zero lies here.",
            "Proving this remains the greatest unsolved mystery in mathematics.",
            "A million-dollar prize awaits whoever solves this puzzle."
        ]
        self.setup_layout(title, lines)

        # Pre-build mobjects
        # 1. Boundaries and Strip
        # Adjusted positions based on Issue 46 and 47
        line_left = Line(self.grid['B1'], self.grid['F1'], color="#808080", stroke_width=2)
        line_right = Line(self.grid['B5'], self.grid['F5'], color="#808080", stroke_width=2)
        
        label_0 = Text("x=0", font_size=16, color="#808080")
        self.place_at_grid(label_0, 'B1', scale_factor=1.0).shift(UP * 0.3)
        
        label_1 = Text("x=1", font_size=16, color="#808080")
        self.place_at_grid(label_1, 'B5', scale_factor=1.0).shift(UP * 0.3)
        
        # Strip moved to row C to avoid overlap (Issue 47)
        strip = Rectangle(width=4, height=3, color="#E6E6FA", fill_opacity=0.3, stroke_width=0)
        self.place_in_area(strip, 'C1', 'F5')
        
        # 2. Critical Line
        critical_line = Line(self.grid['B3'], self.grid['F3'], color="#FFD700", stroke_width=5)
        label_half = Text("x=1/2", font_size=18, color="#FFD700")
        self.place_at_grid(label_half, 'B3', scale_factor=1.0).shift(UP * 0.4)
        
        # 3. Zeros (white dots) - Placed along the critical line within the strip area
        dots = VGroup(*[
            Dot(self.grid[pos], color=WHITE, radius=0.08) 
            for pos in ['C3', 'D3', 'E3', 'F3']
        ])
        # Slight aesthetic adjustments for zeros' spacing
        dots[0].shift(UP * 0.2)
        dots[1].shift(DOWN * 0.3)
        dots[2].shift(UP * 0.1)
        dots[3].shift(DOWN * 0.4)
        
        # 4. Prize text and Asset (Issue 33, 48)
        money_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/money.svg").scale(0.3)
        prize_label = Text("$1,000,000 Prize", font_size=32, color=WHITE, weight=BOLD)
        prize_group = VGroup(money_icon, prize_label).arrange(RIGHT, buff=0.2)
        self.place_in_area(prize_group, 'A1', 'A6')

        # === Animation for Lecture Line 1 ===
        # Color match: #E6E6FA (Purple strip)
        self.play(self.lecture[0].animate.set_color("#E6E6FA"))
        self.play(
            Create(line_left), 
            Create(line_right), 
            FadeIn(strip), 
            Write(label_0), 
            Write(label_1),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color match: #FFD700 (Gold line)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFD700")
        )
        self.play(Create(critical_line), Write(label_half))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color match: #FFFFFF (White zeros)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE)
        )
        self.play(
            LaggedStart(*[FadeIn(dot, scale=0.5) for dot in dots], lag_ratio=0.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Color match: White
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(WHITE)
        )
        self.play(Indicate(critical_line, color="#FFD700"), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Color match: White
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(WHITE)
        )
        self.play(Write(prize_group))
        self.wait(3)
