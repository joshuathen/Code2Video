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
        # Setup layout
        title_text = "Prerequisite: The Concept of Rate of Change"
        lines = [
            "Calculus helps us track changes over time.",
            "Imagine a tank where water flows in and out.",
            "Flow rate depends on the current population size."
        ]
        self.setup_layout(title_text, lines)
        
        # Colors
        line_color = "#FFFF00"  # Yellow for highlight
        water_color = "#3399FF" # Sky Blue
        label_color = "#FFFFFF" # White
        inflow_color = "#00FF00" # Green
        outflow_color = "#FF3333" # Red

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(line_color))
        
        # Tank Asset (Issue 36)
        tank = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/tank.svg")
        tank.set_color(WHITE)
        self.place_in_area(tank, "B3", "E5")
        
        # Water inside (approximate size relative to tank area)
        water = Rectangle(
            width=tank.width * 0.9, 
            height=tank.height * 0.4, 
            fill_color=water_color, 
            fill_opacity=0.6, 
            stroke_width=0
        )
        water.move_to(tank.get_bottom(), aligned_edge=DOWN).shift(UP * 0.1)
        
        # Gauge Label (Issue 41: area C2-E2, scale 0.8)
        gauge_label = Text("Population Size", font_size=18, color=label_color)
        self.place_in_area(gauge_label, "C2", "E2", scale_factor=0.8)
        gauge_label.rotate(PI/2)

        self.play(Create(tank), FadeIn(water), Write(gauge_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(line_color)
        )
        
        # Tap Asset (Issue 36: A3)
        tap = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/tap.svg")
        self.place_at_grid(tap, "A3", scale_factor=0.6)
        
        # Inflow Arrow (From tap to tank)
        in_arrow = Arrow(
            start=self.grid["A3"] + DOWN * 0.3, 
            end=self.grid["B3"] + DOWN * 0.3, 
            color=inflow_color
        )
        in_label = Text("Inflow", font_size=20, color=inflow_color)
        self.place_at_grid(in_label, "A4")
        
        # Outflow Arrow (From tank to E6)
        out_arrow = Arrow(
            start=self.grid["E5"], 
            end=self.grid["E6"], 
            color=outflow_color
        )
        # Outflow Label (Issue 42: E6)
        out_label = Text("Outflow", font_size=20, color=outflow_color)
        self.place_at_grid(out_label, "E6")
        # Shift label slightly so it doesn't overlap arrow head perfectly if needed, 
        # but place_at_grid is the anchor.
        out_label.next_to(out_arrow, RIGHT, buff=0.1)

        self.play(
            FadeIn(tap),
            GrowArrow(in_arrow),
            FadeIn(in_label),
            GrowArrow(out_arrow),
            FadeIn(out_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(line_color)
        )
        
        # Derivative notation label (Issue 40: C6, scale 0.8)
        dv_dt_label = Text("dV / dt", font_size=32, color=label_color)
        self.place_at_grid(dv_dt_label, "C6", scale_factor=0.8)
        
        # Transform labels and simulate volume increase
        self.play(
            ReplacementTransform(VGroup(in_label, out_label), dv_dt_label),
            water.animate.stretch_to_fit_height(tank.height * 0.7, about_edge=DOWN)
        )
        self.wait(2)
        
        # Final cleanup/reset highlight
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
