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
        # Data from shared state
        title = "Prerequisite: The Quantum Canvas"
        lecture_lines = [
            "Quantum computers use superposition to represent all items simultaneously.",
            "We start with every box having an equal probability amplitude.",
            "This state is the starting point for Grover's algorithm."
        ]
        
        self.setup_layout(title, lecture_lines)

        # Colors
        bar_color = "#00FF00"
        glow_color = "#ADD8E6"
        
        # === Elements for Animation ===
        
        # Asset path (Issue 26)
        box_asset = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/box.svg"
        
        # 4 bars using the box SVG (Issue 26)
        # We create them as SVGMobjects to represent the boxes/amplitudes
        bars = VGroup(*[
            SVGMobject(box_asset).set_color(bar_color)
            for _ in range(4)
        ])
        
        # Initial placement: Row E to ensure proximity to labels in Row F (Issue 42 & 43)
        # We start them small (0.6) to represent base probability/state
        for i in range(4):
            self.place_at_grid(bars[i], f"E{i+2}", scale_factor=0.6)
            
        # Labels for boxes in row F (Issue 42 proximity)
        labels = VGroup(
            Text("00", font_size=20),
            Text("01", font_size=20),
            Text("10", font_size=20),
            Text("11", font_size=20)
        )
        for i, label in enumerate(labels):
            self.place_at_grid(label, f"F{i+2}")

        # Math Formula (Issue 41)
        # Fixed: Using Text instead of MathTex to avoid FileNotFoundError: 'latex'
        formula = Text("1/√N Σ |i⟩", font_size=32, color=WHITE)
        self.place_in_area(formula, "B2", "B5", scale_factor=1.2)

        # === Animation for Lecture Line 1 ===
        # Line 1: "Quantum computers use superposition to represent all items simultaneously."
        # Visual: Show 4 box assets with labels '00', '01', '10', '11'.
        self.play(
            self.lecture[0].animate.set_color(YELLOW),
            FadeIn(bars),
            Write(labels),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: "We start with every box having an equal probability amplitude."
        # Visual: Stretch bars to span from Row E up through Row D/C, centering at D (Issue 43)
        # Formula appears at the top (Row B).
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            bars[0].animate.stretch_to_fit_height(2.8).move_to(self.grid["D2"]),
            bars[1].animate.stretch_to_fit_height(2.8).move_to(self.grid["D3"]),
            bars[2].animate.stretch_to_fit_height(2.8).move_to(self.grid["D4"]),
            bars[3].animate.stretch_to_fit_height(2.8).move_to(self.grid["D5"]),
            Write(formula),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: "This state is the starting point for Grover's algorithm."
        # Visual: Add glow effect to all bars.
        
        glows = VGroup(*[
            Rectangle(
                width=bars[i].width + 0.1, 
                height=bars[i].height + 0.1, 
                fill_opacity=0.15, 
                fill_color=glow_color, 
                stroke_width=0
            ).move_to(bars[i].get_center())
            for i in range(4)
        ])

        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW),
            FadeIn(glows, scale=1.1),
            bars.animate.set_stroke(glow_color, opacity=0.6, width=1),
            run_time=2
        )
        self.wait(2)

        # Reset final line color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
