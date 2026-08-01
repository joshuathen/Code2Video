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

class Section1Scene(TeachingScene):
    def construct(self):
        # Data from Snapshot
        title = "The Paradox of Turbulent Flow"
        lines = [
            "Smooth laminar flow appears stable and predictable.",
            "Subtle disturbances trigger oscillations in the fluid stream.",
            "These waves eventually break into rotating golden vortices.",
            "Chaos emerges, yet underlying patterns remain hidden within.",
            "A geometric grid reveals the order inside the chaos."
        ]
        self.setup_layout(title, lines)

        # Colors
        BLUE_C = "#58C4DD"
        GOLD_C = "#F8E71C"
        RED_C = "#FC6255"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE_C)
        
        # Anim 1: A single blue line flows horizontally and smoothly.
        smooth_line = Line(start=LEFT*2.5, end=RIGHT*2.5, color=BLUE_C)
        # Issue 29 Fix: Align with the center of the future wave area (Row C)
        self.place_in_area(smooth_line, 'C1', 'C6', scale_factor=1.0)
        self.play(Create(smooth_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE_C)
        
        # Anim 2: The blue line starts to oscillate into a wave.
        # Issue 28 Fix: Expand area to prevent vertical clipping
        wave = FunctionGraph(
            lambda x: 0.4 * np.sin(3 * PI * x),
            x_range=[-2.5, 2.5],
            color=BLUE_C
        )
        self.place_in_area(wave, 'B1', 'D6', scale_factor=0.8)
        self.play(ReplacementTransform(smooth_line, wave))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GOLD_C)
        
        # Anim 3: The wave breaks into several rotating gold vortices.
        vortices = VGroup()
        # Initial positions for vortices
        chaos_positions = ["B2", "B5", "E2", "E5"]
        for pos in chaos_positions:
            v = VGroup(
                Arc(radius=0.4, angle=TAU*0.7, color=GOLD_C).add_tip(tip_length=0.12),
                Arc(radius=0.2, angle=TAU*0.7, color=GOLD_C).add_tip(tip_length=0.1)
            )
            self.place_at_grid(v, pos)
            vortices.add(v)
            
        self.play(
            ReplacementTransform(wave, vortices),
            run_time=2
        )
        # Swirling effect
        self.play(
            *(Rotating(v, angle=-TAU, run_time=2) for v in vortices)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RED_C)
        
        # Anim 4: A red 'CHAOS' label appears above the vortices.
        chaos_label = Text("CHAOS", color=RED_C, font_size=36)
        # Issue 27 Fix: Center label horizontally in the animation area
        self.place_in_area(chaos_label, 'A2', 'A5', scale_factor=1.2)

        self.play(Write(chaos_label))
        # Continuous movement to emphasize chaos
        self.play(
            *(Rotating(v, angle=-TAU, run_time=3) for v in vortices)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GOLD_C)
        
        # Anim 5: The vortices align to reveal a hidden geometric grid.
        geometric_grid = VGroup()
        # Build a fine grid for reveal
        for col in ["1", "2", "3", "4", "5", "6"]:
            line = Line(self.grid[f"A{col}"], self.grid[f"F{col}"], stroke_width=0.5, color=WHITE, stroke_opacity=0.3)
            geometric_grid.add(line)
        for row in ["A", "B", "C", "D", "E", "F"]:
            line = Line(self.grid[f"{row}1"], self.grid[f"{row}6"], stroke_width=0.5, color=WHITE, stroke_opacity=0.3)
            geometric_grid.add(line)

        # Align vortices to the grid intersections explicitly
        self.play(
            FadeIn(geometric_grid),
            chaos_label.animate.set_opacity(0.3),
            vortices[0].animate.move_to(self.grid["B2"]),
            vortices[1].animate.move_to(self.grid["B5"]),
            vortices[2].animate.move_to(self.grid["E2"]),
            vortices[3].animate.move_to(self.grid["E5"]),
            run_time=2
        )
        
        # Harmonized rotation showing order
        self.play(
            *(Rotating(v, angle=-TAU, run_time=4) for v in vortices)
        )
        self.wait(2)

        # Final cleanup
        self.play(self.lecture.animate.set_color(WHITE))
        self.wait(2)
