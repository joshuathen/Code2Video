from manim import *
import numpy as np

# Base class for maintaining layout and positioning consistency
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
        # Initial Teaching Scene Setup
        lecture_lines_text = [
            "Numbers alone cannot describe a world in motion.",
            "We need a way to capture how things change.",
            "Differential equations are rules for this growth."
        ]
        self.setup_layout("The Hook: Capturing Movement", lecture_lines_text)

        # === Animation for Lecture Line 1 ===
        # Description: Create a cheetah silhouette and a ground line. [Color: #FFFF00]
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Ground line positioned at row F
        ground = Line(self.grid["F1"], self.grid["F6"], color=GREY_B)
        
        # Cheetah SVG asset: [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/cheetah.svg]
        cheetah = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cheetah.svg")
        cheetah.set_color("#FFFF00")
        self.place_at_grid(cheetah, "E1", scale_factor=0.6)
        
        self.play(Create(ground), FadeIn(cheetah))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Description: Move the cheetah across with increasing speed and a velocity vector. [Color: #00FF00]
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Motion boundaries from grid
        start_pos = self.grid["E1"].copy()
        end_pos = self.grid["E6"].copy()
        
        # Velocity arrow setup
        vel_arrow = Arrow(color="#00FF00", buff=0, stroke_width=4)
        vel_arrow.set_opacity(0)
        
        # Tracker for motion (representing normalized time t)
        t_tracker = ValueTracker(0)
        
        # Cheetah updater (s = t^2 for constant acceleration behavior)
        cheetah.add_updater(lambda m: m.move_to(
            start_pos + (end_pos - start_pos) * (t_tracker.get_value()**2)
        ))
        
        # Arrow updater: Grows with velocity (v=2t), stays above cheetah
        def arrow_updater(mob):
            t = t_tracker.get_value()
            if t > 0.05:
                mob.set_opacity(1)
                # Position relative to cheetah top
                base = cheetah.get_top() + UP * 0.15
                # Vector length increases as speed increases
                vec_len = t * 1.2 + 0.1
                mob.put_start_and_end_on(base, base + RIGHT * vec_len)
            else:
                mob.set_opacity(0)

        vel_arrow.add_updater(arrow_updater)
        self.add(vel_arrow)
        
        # Acceleration over time
        self.play(t_tracker.animate.set_value(1), run_time=2.5, rate_func=linear)
        self.wait(1)
        
        # Clear updaters to bake position
        cheetah.clear_updaters()
        vel_arrow.clear_updaters()

        # === Animation for Lecture Line 3 ===
        # Description: Display text 'Rate of Change' and link it to acceleration. [Color: #FFFFFF]
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        # "Rate of Change" concept label
        roc_text = Text("Rate of Change", font_size=22, color=WHITE)
        self.place_at_grid(roc_text, "A3")
        
        # Formal representation (ODE component)
        formula = Text("dv/dt = a", color=WHITE)
        self.place_at_grid(formula, "B3", scale_factor=1.0)
        
        # Target concept link
        accel_label = Text("Acceleration", font_size=22, color=WHITE)
        self.place_at_grid(accel_label, "D5")

        # Connection arrows between concepts
        link_1 = Arrow(self.grid["A3"], self.grid["B3"], color=WHITE, buff=0.4)
        link_2 = Arrow(self.grid["B3"], self.grid["D5"], color=WHITE, buff=0.4)
        
        self.play(
            FadeIn(roc_text),
            Write(formula),
            Create(link_1),
            run_time=1.5
        )
        self.play(
            FadeIn(accel_label),
            Create(link_2),
            run_time=1.2
        )
        
        self.wait(3)