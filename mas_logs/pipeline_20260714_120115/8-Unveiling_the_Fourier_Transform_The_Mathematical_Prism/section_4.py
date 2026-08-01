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
        # Title and lecture lines
        title_str = "The Frequency Spectrum: The Recipe Card"
        lines = [
            "We plot these shifts to create a frequency spectrum.",
            "The horizontal axis represents the test frequencies used.",
            "Vertical bars show the strength of each frequency component.",
            "This spectrum is the recipe card for our signal.",
            "It stays steady even when the signal looks chaotic."
        ]
        self.setup_layout(title_str, lines)
        
        # Colors
        RED_COLOR = "#FF0000"
        GREEN_COLOR = "#00FF00"
        BLUE_COLOR = "#0000FF"
        GLOW_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Line 1: "We plot these shifts to create a frequency spectrum."
        # Animation: Draw a white #FFFFFF horizontal frequency axis.
        self.lecture[0].set_color(WHITE)
        
        # Frequency axis (Horizontal)
        # Using Col 3 to 6 to stay away from Column 1/2 for labels.
        freq_axis = Arrow(
            start=self.grid["E3"], 
            end=self.grid["E6"] + RIGHT*0.2, 
            color=WHITE, 
            buff=0,
            stroke_width=4
        )
        # Strength axis (Vertical)
        amp_axis = Arrow(
            start=self.grid["E3"], 
            end=self.grid["B3"] + UP*0.2, 
            color=WHITE, 
            buff=0,
            stroke_width=4
        )
        
        self.play(Create(freq_axis), Create(amp_axis), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: "The horizontal axis represents the test frequencies used."
        # Animation: Create a tall red #FF0000 spike at a low frequency.
        self.lecture[1].set_color(RED_COLOR)
        
        # Issue 35: Fix horizontal crowding by moving freq_label to F2 and scaling to 0.7
        freq_label = Text("Frequency", font_size=18, color=WHITE)
        self.place_at_grid(freq_label, "F2", scale_factor=0.7) 
        
        red_spike = Line(
            start=self.grid["E4"],
            end=self.grid["B4"],
            color=RED_COLOR,
            stroke_width=8
        )
        # Issue 37: Scale down tags to 0.7
        low_tag = Text("Low", font_size=14, color=RED_COLOR)
        self.place_at_grid(low_tag, "F4", scale_factor=0.7)
        
        self.play(Write(freq_label), Create(red_spike), Write(low_tag), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: "Vertical bars show the strength of each frequency component."
        # Animation: Add a medium green #00FF00 spike at a mid frequency.
        self.lecture[2].set_color(GREEN_COLOR)
        
        # Strength label near the vertical axis (Column 2 is safe)
        amp_label = Text("Strength", font_size=18, color=WHITE).rotate(90*DEGREES)
        self.place_at_grid(amp_label, "C2", scale_factor=0.8)
        
        green_spike = Line(
            start=self.grid["E5"],
            end=self.grid["C5"],
            color=GREEN_COLOR,
            stroke_width=8
        )
        # Issue 37: Scale down tags to 0.7
        mid_tag = Text("Mid", font_size=14, color=GREEN_COLOR)
        self.place_at_grid(mid_tag, "F5", scale_factor=0.7)
        
        self.play(Write(amp_label), Create(green_spike), Write(mid_tag), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line 4: "This spectrum is the recipe card for our signal."
        # Animation: Add a small blue #0000FF spike at a high frequency.
        self.lecture[3].set_color(BLUE_COLOR)
        
        blue_spike = Line(
            start=self.grid["E6"],
            end=self.grid["D6"],
            color=BLUE_COLOR,
            stroke_width=8
        )
        # Issue 37: Scale down tags to 0.7
        high_tag = Text("High", font_size=14, color=BLUE_COLOR)
        self.place_at_grid(high_tag, "F6", scale_factor=0.7)
        
        # "Recipe Card" highlight
        recipe_box = SurroundingRectangle(VGroup(red_spike, green_spike, blue_spike), color=YELLOW, buff=0.2)
        # Issue 36: Fix cramped text by using place_in_area
        recipe_text = Text("Recipe Card", font_size=22, color=YELLOW)
        self.place_in_area(recipe_text, 'A5', 'B6', scale_factor=0.8)
        
        self.play(Create(blue_spike), Write(high_tag), Create(recipe_box), Write(recipe_text), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: "It stays steady even when the signal looks chaotic."
        # Animation: Highlight all spikes with a white #FFFFFF glow.
        self.lecture[4].set_color(GLOW_COLOR)
        
        glows = VGroup(*[
            s.copy().set_stroke(width=15, opacity=0.5).set_color(WHITE)
            for s in [red_spike, green_spike, blue_spike]
        ])
        
        self.play(FadeIn(glows), run_time=1)
        self.play(glows.animate.set_stroke(opacity=0.8), rate_func=there_and_back, run_time=2)
        self.wait(2)
