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
        # Define lecture lines
        lecture_lines = [
            "Imagine a slime moving through a function field.",
            "Local scaling determines the slime's shape and size.",
            "Where f' is 1, the slime stays normal.",
            "Where f' is 0, the slime is crushed thin.",
            "Negative f' values invert the slime's front and back."
        ]
        
        # Colors for lines and corresponding elements
        COLOR_L1 = WHITE
        COLOR_L2 = "#FFFF00"  # Yellow
        COLOR_L3 = "#88FF88"  # Green
        COLOR_L4 = "#FF8888"  # Red
        COLOR_L5 = "#FFCC88"  # Orange
        FLASH_COLOR = "#FFFF00" # Yellow
        
        self.setup_layout("Application: The Stretchy Slime", lecture_lines)
        
        # Initialize line colors (all white initially)
        for line in self.lecture:
            line.set_color(WHITE)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_L1)
        
        # Create input line (horizontal axis)
        input_line = Line(self.grid['D1'], self.grid['D6'], color="#555555")
        self.add(input_line)
        
        # Create Slime using Asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/slime.svg]
        slime_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/slime.svg"
        slime = SVGMobject(slime_asset_path)
        # Store original points for non-destructive stretching in updater
        # Scale to a base size first
        slime.scale(0.5)
        slime.original_points = slime.points.copy()
        
        # ValueTracker for x (starts slightly before Col 2)
        x_tracker = ValueTracker(-PI/4)
        
        # Optimized updater that modifies geometry and position in place
        def update_slime(m):
            x = x_tracker.get_value()
            # Mapping x=[0, PI] to grid_x=[1.5, 5.5] (roughly Cols 2 to 6)
            gx = 1.5 + (x / PI) * 4.0
            gy = self.grid['D1'][1] # Row D
            
            # Derivative value s = f'(x) = cos(x) for f(x)=sin(x)
            s = np.cos(x)
            
            # Reset to base and stretch
            m.points = m.original_points.copy()
            # Stretch horizontally by s. If s is small, keep a minimum width for visibility.
            # If s < 0, it flips horizontally.
            stretch_factor = s if abs(s) > 0.05 else (0.05 if s >= 0 else -0.05)
            m.stretch(stretch_factor, dim=0)
            m.move_to([gx, gy, 0])

        slime.add_updater(update_slime)
        
        # Initial reveal
        self.play(FadeIn(slime))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_L2))
        
        scaling_label = MathTex("f'(x)", font_size=32, color=COLOR_L2)
        # Fix for Issue 29: Place in area A3-A4
        self.place_in_area(scaling_label, 'A3', 'A4')
        self.play(Write(scaling_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_L3))
        
        # Move to x=0 (f'(0)=1)
        self.play(x_tracker.animate.set_value(0), run_time=1.5, rate_func=linear)
        
        val_1 = MathTex("f'(0) = 1", font_size=24, color=COLOR_L3)
        # Fix for Issue 30: Place in area B1-B2
        self.place_in_area(val_1, 'B1', 'B2')
        self.play(Write(val_1))
        self.play(Flash(val_1, color=FLASH_COLOR, run_time=0.5))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_L4))
        
        # Move to x=PI/2 (f'(PI/2)=0)
        self.play(x_tracker.animate.set_value(PI/2), run_time=2, rate_func=linear)
        
        val_2 = MathTex("f'(\\pi/2) = 0", font_size=24, color=COLOR_L4)
        # Fix for Issue 30: Place in area B3-B4
        self.place_in_area(val_2, 'B3', 'B4')
        self.play(Write(val_2))
        self.play(Flash(val_2, color=FLASH_COLOR, run_time=0.5))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_L5))
        
        # Move to x=PI (f'(PI)=-1)
        self.play(x_tracker.animate.set_value(PI), run_time=2, rate_func=linear)
        
        val_3 = MathTex("f'(\\pi) = -1", font_size=24, color=COLOR_L5)
        # Fix for Issue 30: Place in area B5-B6
        self.place_in_area(val_3, 'B5', 'B6')
        self.play(Write(val_3))
        self.play(Flash(val_3, color=FLASH_COLOR, run_time=0.5))
        
        # Cleanup updaters at end of section
        slime.clear_updaters()
        self.wait(2)
