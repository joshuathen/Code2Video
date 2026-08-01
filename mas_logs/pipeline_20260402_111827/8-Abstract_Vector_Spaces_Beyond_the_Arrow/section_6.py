from manim import *
import numpy as np
import pathlib

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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "The Power of Abstraction"
        lines = [
            "Abstract theorems apply to functions, signals, and matrices.",
            "One proof unlocks many fields of science simultaneously.",
            "This is the ultimate power of mathematical abstraction."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Colors
        COLOR_SIGNALS = "#00FFFF"
        COLOR_MATRICES = "#00FF00"
        COLOR_FUNCTIONS = "#FF00FF"
        COLOR_KEY = "#FFD700"

        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(COLOR_SIGNALS))

        # Create Icons
        # Signal Wave Icon (Asset)
        signals_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/wave.svg")
        signals_icon.set_color(COLOR_SIGNALS)
        self.place_at_grid(signals_icon, "B2", scale_factor=0.6)
        signals_label = Text("Signals", font_size=20, color=COLOR_SIGNALS)
        self.place_at_grid(signals_label, "C2", scale_factor=0.8)

        # Matrix Grid Icon (Custom built)
        matrices_icon = VGroup()
        for i in range(4):
            matrices_icon.add(Line([-0.4, i/3-0.5, 0], [0.4, i/3-0.5, 0], stroke_width=2))
            matrices_icon.add(Line([i/3-0.5, -0.4, 0], [i/3-0.5, 0.4, 0], stroke_width=2))
        matrices_icon.set_color(COLOR_MATRICES)
        self.place_at_grid(matrices_icon, "B4", scale_factor=0.6)
        matrices_label = Text("Matrices", font_size=20, color=COLOR_MATRICES)
        self.place_at_grid(matrices_label, "C4", scale_factor=0.8)

        # Function Curve Icon (Custom built)
        functions_icon = ParametricFunction(
            lambda t: np.array([t, 0.4*np.sin(PI*t), 0]), 
            t_range=[-1, 1], 
            color=COLOR_FUNCTIONS
        )
        self.place_at_grid(functions_icon, "B6", scale_factor=0.6)
        functions_label = Text("Functions", font_size=20, color=COLOR_FUNCTIONS)
        self.place_at_grid(functions_label, "C6", scale_factor=0.8)

        # Animation Group 1: Show icons and their labels
        self.play(
            FadeIn(signals_icon), Write(signals_label),
            FadeIn(matrices_icon), Write(matrices_label),
            FadeIn(functions_icon), Write(functions_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_KEY)
        )

        # Create Key Icon (Asset)
        key_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/key.svg")
        key_icon.set_color(COLOR_KEY)
        key_label = Text("Abstract Proof", font_size=18, color=COLOR_KEY)
        
        # Start key off-screen or at left grid
        self.place_at_grid(key_icon, "B1", scale_factor=0.5)
        key_label.add_updater(lambda d: d.next_to(key_icon, UP, buff=0.1))
        
        self.add(key_icon, key_label)
        
        # Key movement sequence with "glow" (Indicate)
        for pos, icon, color in [("B2", signals_icon, COLOR_SIGNALS), 
                                 ("B4", matrices_icon, COLOR_MATRICES), 
                                 ("B6", functions_icon, COLOR_FUNCTIONS)]:
            self.play(key_icon.animate.move_to(self.grid[pos]), run_time=0.8)
            self.play(Indicate(icon, color=color, scale_factor=1.3), run_time=0.5)

        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE) 
        )

        # Morph targets
        qm_label = Text("Quantum Mechanics", font_size=18, color=WHITE)
        cg_label = Text("Computer Graphics", font_size=18, color=WHITE)
        ds_label = Text("Data Science", font_size=18, color=WHITE)
        
        self.place_at_grid(qm_label, "C2")
        self.place_at_grid(cg_label, "C4")
        self.place_at_grid(ds_label, "C6")

        # Morph icons and current labels into application labels
        self.play(
            FadeOut(key_icon),
            key_label.animate.set_opacity(0),
            ReplacementTransform(VGroup(signals_icon, signals_label), qm_label),
            ReplacementTransform(VGroup(matrices_icon, matrices_label), cg_label),
            ReplacementTransform(VGroup(functions_icon, functions_label), ds_label),
            run_time=1.5
        )
        key_label.clear_updaters()

        # Zoom out effect (Encompass everything)
        apps_group = VGroup(qm_label, cg_label, ds_label)
        
        # Box representing the general theory
        enclosure_box = RoundedRectangle(corner_radius=0.2, color=WHITE)
        enclosure_box.surround(apps_group, stretch=True, buff=0.4)
        
        la_header = Text("Linear Algebra Theorems", font_size=24, color=WHITE)
        la_header.next_to(enclosure_box, UP, buff=0.2)
        
        # Simulate zoom-out by scaling the group down slightly and adding the boundary
        self.play(
            apps_group.animate.scale(0.8),
            run_time=1
        )
        self.play(
            Create(enclosure_box),
            Write(la_header),
            run_time=1
        )
        
        self.wait(3)
