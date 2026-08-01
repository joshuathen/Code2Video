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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup the layout with mandatory title and lecture lines
        self.setup_layout("Anatomy of a Nut: The Hard Truth", [
            "True nuts are dry fruits with one seed.",
            "Their ovary walls become stony and very hard.",
            "They are indehiscent, meaning they stay closed."
        ])

        # === Animation for Lecture Line 1 ===
        # A brown acorn (#8B4513) appears beside a red 'Hardness Meter' (#FF0000).
        self.lecture[0].set_color("#8B4513")
        
        # Acorn Graphic
        acorn_body = Ellipse(width=1.0, height=1.2, fill_color="#8B4513", fill_opacity=1, stroke_width=0)
        acorn_cap = AnnularSector(
            inner_radius=0, 
            outer_radius=0.6, 
            angle=PI, 
            start_angle=0, 
            fill_color="#5D2E0C", 
            fill_opacity=1
        ).shift(UP * 0.25)
        acorn = VGroup(acorn_body, acorn_cap)
        # Fix Issue 43: Reposition acorn
        self.place_in_area(acorn, 'B3', 'D4', scale_factor=1.2)

        # Hardness Meter
        meter_frame = Rectangle(height=2.0, width=0.4, color=WHITE)
        meter_fill = Rectangle(
            height=0.05, 
            width=0.3, 
            fill_color="#FF0000", 
            fill_opacity=1, 
            stroke_width=0
        ).align_to(meter_frame, DOWN).shift(UP * 0.05)
        meter_label = Text("Hardness", font_size=16, color=WHITE)
        hardness_meter = VGroup(meter_frame, meter_fill)
        # Fix Issue 44: Reposition hardness meter and label
        self.place_in_area(hardness_meter, 'B5', 'D5', scale_factor=1.0)
        self.place_at_grid(meter_label, 'A5', scale_factor=1.0)

        self.play(FadeIn(acorn), FadeIn(hardness_meter), Write(meter_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The meter rises to 'Stony' as the acorn's shell (#FFFFFF) glows.
        self.lecture[1].set_color("#FF0000")
        
        stony_text = Text("STONY", font_size=18, color="#FF0000", weight=BOLD)
        # Fix Issue 44: Reposition stony text
        self.place_at_grid(stony_text, 'B6', scale_factor=1.0)
        
        # Glow effect
        glow_shell = acorn_body.copy().set_fill(opacity=0).set_stroke(color="#FFFFFF", width=6)
        
        self.play(
            meter_fill.animate.stretch_to_fit_height(1.8).align_to(meter_frame, DOWN).shift(UP * 0.05),
            Write(stony_text),
            Create(glow_shell),
            run_time=2
        )
        self.play(FadeOut(glow_shell))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The word 'Indehiscent' (#00BFFF) appears, indicating the shell does not open.
        self.lecture[2].set_color("#00BFFF")
        
        indehiscent_label = Text("Indehiscent", font_size=32, color="#00BFFF")
        # Fix Issue 43: Reposition indehiscent label and adjust scale
        self.place_at_grid(indehiscent_label, 'E3', scale_factor=0.7)
        
        # Lock icon
        lock_base = Rectangle(width=0.4, height=0.3, fill_color=WHITE, fill_opacity=1)
        lock_handle = Arc(radius=0.15, angle=PI, color=WHITE, stroke_width=4).shift(UP * 0.15)
        lock = VGroup(lock_base, lock_handle)
        # Fix Issue 43: Reposition lock and adjust scale
        self.place_at_grid(lock, 'E4', scale_factor=0.8)

        self.play(
            Write(indehiscent_label),
            FadeIn(lock)
        )
        self.wait(1)

        # === Final Comparison (Oranges vs Nuts) ===
        # Add a "Juice" meter for comparison (Fixing Issue 42)
        juice_frame = Rectangle(height=2.0, width=0.4, color=WHITE)
        juice_fill = Rectangle(
            height=1.8, 
            width=0.3, 
            fill_color="#FFA500", 
            fill_opacity=0.8, 
            stroke_width=0
        ).align_to(juice_frame, DOWN).shift(UP * 0.05)
        juice_label = Text("Juice", font_size=16, color="#FFA500")
        juice_meter = VGroup(juice_frame, juice_fill)
        
        # Fix Issue 42: Positioning the comparison juice meter
        self.place_in_area(juice_meter, 'B2', 'D2', scale_factor=1.0)
        self.place_at_grid(juice_label, 'A2', scale_factor=1.0)
        
        cross = VGroup(
            Line(LEFT, RIGHT, color=RED, stroke_width=8).rotate(45*DEGREES),
            Line(LEFT, RIGHT, color=RED, stroke_width=8).rotate(-45*DEGREES)
        ).scale(0.3)
        # Fix Issue 42: Positioning the cross
        self.place_at_grid(cross, 'C2', scale_factor=1.0)

        self.play(FadeIn(juice_meter), Write(juice_label))
        self.play(Create(cross))
        self.wait(2)
