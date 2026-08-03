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
        # Section 2: Prerequisite: Wave Interference and Coherence
        lecture_lines = [
            "- Holography requires coherent laser light.",
            "- Coherent waves oscillate in perfect sync.",
            "- Superposition creates stationary interference patterns."
        ]
        self.setup_layout("Prerequisite: Wave Interference and Coherence", lecture_lines)
        
        # Colors for matching lecture lines
        color_wave = "#00FF00"
        color_ripples = "#ADD8E6"
        
        # === Animation for Lecture Line 1 ===
        # Step 1: Show two sine waves (#00FF00) perfectly in phase, originating from a laser.
        self.lecture[0].set_color(color_wave)
        
        # Load laser asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg]
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg", color=WHITE)
        self.place_at_grid(laser, 'B1', scale_factor=0.6)
        
        axes1 = Axes(x_range=[-2, 2], y_range=[-1.5, 1.5], axis_config={"include_ticks": False, "stroke_width": 1})
        axes2 = Axes(x_range=[-2, 2], y_range=[-1.5, 1.5], axis_config={"include_ticks": False, "stroke_width": 1})
        
        wave1 = axes1.plot(lambda x: np.sin(2 * PI * x), color=color_wave)
        wave2 = axes2.plot(lambda x: np.sin(2 * PI * x), color=color_wave)
        
        group1 = VGroup(axes1, wave1)
        group2 = VGroup(axes2, wave2)
        
        self.place_in_area(group1, 'A1', 'A6', scale_factor=0.6)
        self.place_in_area(group2, 'C1', 'C6', scale_factor=0.6) # Issue 28: use row C
        
        self.play(FadeIn(laser))
        self.play(Create(group1), Create(group2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Step 2: Merge waves to show a larger peak (constructive).
        self.lecture[1].set_color(color_wave)
        
        # Define constructive interference result
        axes_c = Axes(x_range=[-2, 2], y_range=[-2.5, 2.5], axis_config={"include_ticks": False, "stroke_width": 1})
        wave_c = axes_c.plot(lambda x: 2 * np.sin(2 * PI * x), color=color_wave)
        group_c = VGroup(axes_c, wave_c)
        self.place_in_area(group_c, 'A1', 'C6', scale_factor=0.6) # Issue 27: use A1-C6
        
        # Move both to the same area and transform
        self.play(
            group1.animate.move_to(group_c.get_center()),
            group2.animate.move_to(group_c.get_center()),
        )
        self.play(ReplacementTransform(VGroup(group1, group2), group_c))
        self.wait(1)
        
        # Step 3: Shift one wave and merge to show flatline (destructive).
        axes3 = Axes(x_range=[-2, 2], y_range=[-1.5, 1.5], axis_config={"include_ticks": False, "stroke_width": 1})
        axes4 = Axes(x_range=[-2, 2], y_range=[-1.5, 1.5], axis_config={"include_ticks": False, "stroke_width": 1})
        wave3 = axes3.plot(lambda x: np.sin(2 * PI * x), color=color_wave)
        wave4 = axes4.plot(lambda x: np.sin(2 * PI * x + PI), color=color_wave) # Phase shift by PI
        
        group3 = VGroup(axes3, wave3)
        group4 = VGroup(axes4, wave4)
        
        self.place_in_area(group3, 'A1', 'A6', scale_factor=0.6)
        self.place_in_area(group4, 'C1', 'C6', scale_factor=0.6) # Issue 28: use row C
        
        self.play(ReplacementTransform(group_c, VGroup(group3, group4)))
        self.wait(1)
        
        # Destructive result
        axes_d = Axes(x_range=[-2, 2], y_range=[-1.5, 1.5], axis_config={"include_ticks": False, "stroke_width": 1})
        line_d = Line(start=axes_d.c2p(-2, 0), end=axes_d.c2p(2, 0), color=color_wave)
        group_d = VGroup(axes_d, line_d)
        self.place_in_area(group_d, 'A1', 'C6', scale_factor=0.6) # Issue 27: use A1-C6
        
        self.play(
            group3.animate.move_to(group_d.get_center()),
            group4.animate.move_to(group_d.get_center()),
        )
        self.play(ReplacementTransform(VGroup(group3, group4), group_d))
        self.wait(1)
        self.play(FadeOut(group_d), FadeOut(laser))

        # === Animation for Lecture Line 3 ===
        # Step 4: Show overlapping circular ripples (#ADD8E6) from a top view.
        # Step 5: Highlight the static 'fringe' pattern where ripples intersect.
        self.lecture[2].set_color(color_ripples)
        
        source1_dot = Dot(color=WHITE)
        source2_dot = Dot(color=WHITE)
        self.place_at_grid(source1_dot, 'D2')
        self.place_at_grid(source2_dot, 'D5')
        
        # Ripples represented by concentric circles
        ripples1 = VGroup(*[Circle(radius=r, color=color_ripples, stroke_opacity=max(0, 1-r/3)).move_to(source1_dot.get_center()) for r in np.arange(0.3, 2.6, 0.4)])
        ripples2 = VGroup(*[Circle(radius=r, color=color_ripples, stroke_opacity=max(0, 1-r/3)).move_to(source2_dot.get_center()) for r in np.arange(0.3, 2.6, 0.4)])
        
        self.play(FadeIn(source1_dot), FadeIn(source2_dot))
        self.play(Create(ripples1), Create(ripples2), run_time=2)
        
        # Highlight static 'fringe' pattern (areas of intersection)
        # We use a grid-based area to show the static pattern.
        fringes = VGroup(
            Dot(self.grid['C3'], color=YELLOW, radius=0.1),
            Dot(self.grid['D3'], color=YELLOW, radius=0.1),
            Dot(self.grid['E3'], color=YELLOW, radius=0.1),
            Dot(self.grid['C4'], color=YELLOW, radius=0.1),
            Dot(self.grid['D4'], color=YELLOW, radius=0.1),
            Dot(self.grid['E4'], color=YELLOW, radius=0.1)
        )
        
        fringe_label = Text("Interference Fringes", font_size=20, color=YELLOW)
        self.place_in_area(fringe_label, 'F2', 'F5') # Issue 26
        
        self.play(FadeIn(fringes), Write(fringe_label))
        self.play(Indicate(fringes))
        self.wait(2)
