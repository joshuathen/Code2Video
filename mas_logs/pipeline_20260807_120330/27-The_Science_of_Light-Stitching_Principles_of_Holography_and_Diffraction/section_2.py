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
        # Setup title and lecture lines
        title_text = "Prerequisite: The Dance of Waves (Interference & Phase)"
        lecture_lines = [
            "Waves interact through a process called interference.",
            "Constructive interference occurs when wave crests align perfectly.",
            "Destructive interference happens when crests meet wave troughs."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors from storyboard
        color_line1 = "#FFFFFF"
        color_line2 = "#00FF00"
        color_line3 = "#FF00FF"
        color_ripples = "#FFFFFF"
        color_pattern = "#FFFF00"

        # Asset path
        pebble_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/pebb.svg"

        # Helper for sine wave points (performance optimized)
        def get_wave_points(phase, x_width=2.5, amplitude=0.4):
            x_vals = np.linspace(-x_width/2, x_width/2, 50)
            return [np.array([x, amplitude * np.sin(2 * PI * (x - phase)), 0]) for x in x_vals]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_line1)
        
        pebble = SVGMobject(pebble_path).scale(0.4)
        self.place_at_grid(pebble, "C5") # Issue 36: pebble shifted to C5 for balance
        
        ripple1 = Circle(radius=0.1, color=color_ripples).move_to(self.grid["C4"])
        ripple2 = Circle(radius=0.1, color=color_ripples).move_to(self.grid["C6"])
        
        # Interference Pattern dots
        pattern = VGroup()
        for r in ["B", "C", "D", "E"]:
            for c in ["4", "5", "6"]:
                dot = Dot(color=color_pattern, radius=0.06).move_to(self.grid[f"{r}{c}"])
                dist = np.linalg.norm(self.grid[f"{r}{c}"] - self.grid["C5"])
                dot.set_opacity(np.clip(np.abs(np.cos(dist * 3)), 0.2, 0.8))
                pattern.add(dot)

        self.play(FadeIn(pebble))
        self.play(
            ripple1.animate.scale(10).set_stroke(opacity=0),
            ripple2.animate.scale(10).set_stroke(opacity=0),
            FadeIn(pattern),
            run_time=2
        )
        self.wait(1)
        self.play(FadeOut(pebble), FadeOut(ripple1), FadeOut(ripple2), FadeOut(pattern))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color_line2)
        
        label_const = Text("Constructive", font_size=22, color=color_line2)
        self.place_at_grid(label_const, "A5") # Issue 34: label shifted to A5
        
        w2a = VMobject(color=color_line2).set_points_as_corners(get_wave_points(0))
        w2b = VMobject(color=color_line2).set_points_as_corners(get_wave_points(0))
        wave_group_const = VGroup(w2a, w2b).arrange(DOWN, buff=0.8)
        
        # Issue 35: Explicit area containment for wave group
        self.place_in_area(wave_group_const, 'B4', 'F6') 
        
        sum_wave = VMobject(color=color_line2, stroke_width=5).set_points_as_corners(
            get_wave_points(0, amplitude=0.8)
        ).move_to(wave_group_const.get_center())
        
        self.play(Create(wave_group_const), Write(label_const))
        self.wait(0.5)
        # Animate waves meeting at their shared group center
        self.play(w2a.animate.move_to(w2b.get_center()), run_time=1)
        self.play(ReplacementTransform(VGroup(w2a, w2b), sum_wave))
        self.wait(1.5)
        self.play(FadeOut(sum_wave), FadeOut(label_const))
        self.lecture[1].set_color(WHITE)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(color_line3)
        
        label_dest = Text("Destructive", font_size=22, color=color_line3)
        self.place_at_grid(label_dest, "A5") # Issue 34: label shifted to A5
        
        w3a = VMobject(color=color_line3).set_points_as_corners(get_wave_points(0))
        w3b = VMobject(color=color_line3).set_points_as_corners(get_wave_points(0.5)) # 180 deg phase shift
        wave_group_dest = VGroup(w3a, w3b).arrange(DOWN, buff=0.8)
        
        # Issue 35: Explicit area containment for wave group
        self.place_in_area(wave_group_dest, 'B4', 'F6') 
        
        flat_line = Line(LEFT*1.25, RIGHT*1.25, color=color_line3, stroke_width=4).move_to(wave_group_dest.get_center())
        
        self.play(Create(wave_group_dest), Write(label_dest))
        self.wait(0.5)
        # Animate waves meeting
        self.play(w3a.animate.move_to(w3b.get_center()), run_time=1)
        self.play(ReplacementTransform(VGroup(w3a, w3b), flat_line))
        self.wait(2)
        self.play(FadeOut(flat_line), FadeOut(label_dest))
        self.lecture[2].set_color(WHITE)
