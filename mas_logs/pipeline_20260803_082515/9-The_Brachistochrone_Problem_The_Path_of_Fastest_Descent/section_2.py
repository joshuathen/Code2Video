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

class Section2Scene(TeachingScene):
    def construct(self):
        # Fetching data from storyboard
        section_title = "Prerequisite Knowledge: Speed and Energy"
        lecture_lines = [
            "Gravity converts potential energy into kinetic energy.",
            "Vertical drop determines the final speed reached.",
            "Total time depends on speed and distance traveled."
        ]
        
        self.setup_layout(section_title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color(YELLOW)
        
        # Display the energy formula 'mgh = 1/2 mv^2' at the top center in white (#FFFFFF).
        # Fix Issue 31: Move formula1 to 'A4' to 'B6' and scale 0.8
        formula1 = MathTex("mgh = \\frac{1}{2} mv^2", color=WHITE)
        self.place_in_area(formula1, "A4", "B6", scale_factor=0.8)
        
        self.play(Write(formula1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update highlighting
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        
        # Vertical drop setup
        start_point = self.grid["B2"]
        end_point = self.grid["F2"]
        drop_line = DashedLine(start_point, end_point, color=GRAY)
        h_label = MathTex("h", color=GRAY).next_to(drop_line, LEFT, buff=0.2)
        
        # Issue 24: Use marble asset
        marble = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/marble.svg")
        marble.set_color(WHITE)
        marble.scale(0.2)
        marble.move_to(start_point)
        
        # Speed meter at D4 area
        meter_bg = Rectangle(height=2.5, width=0.5, color=WHITE, stroke_width=2)
        self.place_at_grid(meter_bg, "D4")
        meter_label = Text("Speed", font_size=18, color="#00FF00").next_to(meter_bg, UP, buff=0.2)
        
        # Meter fill
        speed_bar = Rectangle(
            width=0.45, 
            height=0.01, 
            fill_color="#00FF00", 
            fill_opacity=0.8, 
            stroke_width=0
        ).move_to(meter_bg.get_bottom(), aligned_edge=DOWN)
        
        fall_tracker = ValueTracker(0) # 0 to 1
        
        # Updaters
        def marble_updater(m):
            m.move_to(interpolate(start_point, end_point, fall_tracker.get_value()))
            
        def bar_updater(m):
            # v = sqrt(2gh) => v is proportional to sqrt(fall_progress)
            v_ratio = np.sqrt(fall_tracker.get_value())
            # Map v_ratio to bar height (max height is roughly the meter_bg height)
            new_height = max(0.01, v_ratio * 2.4)
            m.stretch_to_fit_height(new_height)
            m.move_to(meter_bg.get_bottom(), aligned_edge=DOWN)

        marble.add_updater(marble_updater)
        speed_bar.add_updater(bar_updater)
        
        self.add(drop_line, h_label, marble, meter_bg, meter_label, speed_bar)
        
        # Simulation
        # Use ease_in_quad rate func to simulate gravity-like acceleration
        self.play(fall_tracker.animate.set_value(1), run_time=3, rate_func=rate_functions.ease_in_quad)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update highlighting
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Show the time integral 't = integral(ds/v)' and highlight 'v' in green (#00FF00) and 'ds' in yellow (#FFD700).
        # Fix Issue 32: Position formula2 at E5-F6 with scale factor 1.0
        formula2 = MathTex(
            "t", "=", "\\int", "{ds", "\\over", "v}",
            color=WHITE
        )
        # Apply colors to specific parts
        formula2.set_color_by_tex("ds", "#FFD700")
        formula2.set_color_by_tex("v", "#00FF00")
        
        self.place_in_area(formula2, "E5", "F6", scale_factor=1.0)
        
        self.play(FadeIn(formula2, shift=UP))
        self.wait(3)
