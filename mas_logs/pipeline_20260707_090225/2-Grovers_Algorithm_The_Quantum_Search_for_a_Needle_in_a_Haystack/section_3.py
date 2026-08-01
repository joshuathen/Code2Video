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

class Section3Scene(TeachingScene):
    def construct(self):
        # Colors
        YELLOW_C = "#FFFF00"
        GOLD_C = "#FFD700"
        CYAN_C = "#00FFFF"
        
        lines = [
            'The Oracle is a function that identifies our target.',
            'It recognizes the correct item without revealing its location.',
            "Mathematically, the Oracle flips the target's phase.",
            "The target's amplitude points downward on our chart.",
            'Its probability remains unchanged at this specific stage.'
        ]
        
        self.setup_layout("Step 1: The Oracle (Phase Inversion)", lines)
        
        # --- Persistent Components ---
        # Bar Chart Setup
        axes = Axes(
            x_range=[0, 9, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=4.5,
            y_length=4,
            axis_config={"include_tip": False, "color": GREY},
            tips=False
        )
        # Fix Issue 28: Scale axes by 0.8 to avoid crowding vertical space
        self.place_in_area(axes, "B2", "E6", scale_factor=0.8)
        
        # Bars initialization (8 bars)
        base_height = 1.2
        bars = VGroup()
        for i in range(1, 9):
            bar = Rectangle(
                width=0.3,
                height=base_height,
                fill_opacity=0.8,
                color=BLUE_D,
                stroke_width=1
            )
            # Ensure bars match the 0.8 scaling of the axes
            bar.scale(0.8)
            bar.move_to(axes.c2p(i, base_height/2))
            bars.add(bar)
            
        target_index = 3 # Index 0-7, corresponds to bar at x=4
        target_bar = bars[target_index]

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW_C))
        self.play(Create(axes), Create(bars))
        self.play(target_bar.animate.set_color(YELLOW_C))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW_C))
        self.play(Indicate(target_bar, color=YELLOW_C))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(CYAN_C))
        
        # Replacement for MathTex due to environment limitations (missing latex)
        formula = Text("|w⟩ → -|w⟩", color=CYAN_C, font_size=24)
        # Fix Issue 27: Match scale with probability formula (0.8)
        self.place_at_grid(formula, "A4", scale_factor=0.8)
        
        self.play(
            target_bar.animate.rotate(PI, axis=RIGHT, about_point=axes.c2p(4, 0)),
            Write(formula)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(GOLD_C))
        
        # Amplitude Label using Text instead of MathTex
        label = Text("-1/√N", color=GOLD_C, font_size=24)
        label.next_to(target_bar, DOWN, buff=0.2)
        
        self.play(Write(label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(GOLD_C))
        
        # Probability formula using Text instead of MathTex
        prob_formula = Text("|-a|² = |a|²", color=WHITE, font_size=24)
        # Fix Issue 29: Move to A3 to better balance the header row
        self.place_at_grid(prob_formula, "A3", scale_factor=0.8)
        
        self.play(FadeIn(prob_formula))
        self.play(Indicate(target_bar))
        self.wait(2)
