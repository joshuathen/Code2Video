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
        # Description: Create a [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/cheetah.svg] silhouette and a ground line. [Color: #FFFF00]
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Ground line positioned at row F
        ground_start = self.grid["F1"] + LEFT * 0.5 + DOWN * 0.4
        ground_end = self.grid["F6"] + RIGHT * 0.5 + DOWN * 0.4
        ground = Line(ground_start, ground_end, color=GREY_B)
        
        # Cheetah SVG asset
        cheetah = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cheetah.svg")
        cheetah.set_color("#FFFF00")
        self.place_at_grid(cheetah, "F1", scale_factor=0.6)
        
        self.play(Create(ground), FadeIn(cheetah))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Description: Move the [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/cheetah.svg] across with increasing speed and a velocity vector. [Color: #00FF00]
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Velocity arrow setup (Green to match lecture line color)
        vel_arrow = Arrow(LEFT, RIGHT, color="#00FF00", buff=0, stroke_width=5)
        vel_arrow.set_width(0.1)
        
        # ValueTracker for movement (0 to 1)
        tracker = ValueTracker(0)
        start_pt = self.grid["F1"]
        end_pt = self.grid["F6"]
        
        # Use updaters for smooth motion and growth
        def update_cheetah(mob):
            t = tracker.get_value()
            # Quadratic position s(t) = s0 + (s1-s0)*t^2 for constant acceleration
            mob.move_to(start_pt + (end_pt - start_pt) * (t**2))
            
        def update_arrow(mob):
            t = tracker.get_value()
            # Position relative to current cheetah calculation
            current_pos = start_pt + (end_pt - start_pt) * (t**2)
            mob.move_to(current_pos + UP * 0.5, aligned_edge=LEFT)
            # Width represents velocity (derivative of t^2 is 2t)
            new_width = max(0.1, t * 1.8)
            mob.set_width(new_width, stretch=True, about_edge=LEFT)

        cheetah.add_updater(update_cheetah)
        vel_arrow.add_updater(update_arrow)
        
        self.add(vel_arrow)
        self.play(tracker.animate.set_value(1), run_time=2.0, rate_func=linear)
        
        cheetah.clear_updaters()
        vel_arrow.clear_updaters()
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Description: Display text 'Rate of Change' and link it to the acceleration of the [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/cheetah.svg]. [Color: #FFFFFF]
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        # Labels and formulas
        roc_text = Text("Rate of Change", font_size=22, color="#FFFFFF")
        self.place_at_grid(roc_text, "B3")
        
        formula = MathTex(r"\frac{dv}{dt} = a", color="#FFFFFF")
        self.place_at_grid(formula, "C3", scale_factor=1.2)
        
        # Conceptual label for acceleration
        accel_label = Text("Acceleration", font_size=18, color="#FFFFFF")
        self.place_at_grid(accel_label, "E6")

        # Visual links using grid coordinates to maintain strictly valid placement
        link_1 = Arrow(self.grid["B3"] + DOWN * 0.3, self.grid["C3"] + UP * 0.3, color=WHITE, buff=0.1)
        link_2 = Arrow(self.grid["C3"] + DOWN * 0.4, self.grid["E6"] + LEFT * 0.6 + UP * 0.3, color=WHITE, buff=0.1)
        
        self.play(Write(roc_text))
        self.play(Create(link_1), FadeIn(formula))
        self.play(Create(link_2), FadeIn(accel_label))
        
        self.wait(2)
