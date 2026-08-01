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
        
        # Ground line spanning the right side bottom grid row
        ground = Line(
            start=self.grid["F1"] + LEFT*0.5 + DOWN*0.4, 
            end=self.grid["F6"] + RIGHT*0.5 + DOWN*0.4, 
            color=GREY_B
        )
        
        # Load Cheetah SVG asset
        cheetah = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cheetah.svg")
        cheetah.set_color("#FFFF00")
        self.place_at_grid(cheetah, "F1", scale_factor=0.6)
        
        self.play(Create(ground))
        self.play(FadeIn(cheetah))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Description: Move the [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/cheetah.svg] across with increasing speed and a velocity vector. [Color: #00FF00]
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Setup indicators
        vel_arrow = Arrow(LEFT, RIGHT, color="#00FF00", buff=0, stroke_width=4)
        accel_label = Text("a = constant", font_size=16, color="#00FF00")
        
        # Initial placement
        vel_arrow.next_to(cheetah, UP, buff=0.1)
        accel_label.next_to(cheetah, DOWN, buff=0.1)
        
        # Animation parameters using grid references
        start_x = self.grid["F1"][0]
        end_x = self.grid["F6"][0]
        distance = end_x - start_x
        
        t_tracker = ValueTracker(0)

        # Efficient Updaters
        def update_cheetah(mob):
            t = t_tracker.get_value()
            # x = x0 + 0.5 * a * t^2 -> simplified to x0 + dist * t^2
            new_x = start_x + distance * (t**2)
            mob.set_x(new_x)

        def update_vel_arrow(mob):
            t = t_tracker.get_value()
            # Velocity grows linearly with time: v = a * t
            # Map t=[0,1] to arrow width=[0.1, 1.5]
            v_width = 0.1 + 1.4 * t
            mob.set_width(v_width, stretch=True, about_edge=LEFT)
            mob.next_to(cheetah, UP, buff=0.1)

        def update_accel_label(mob):
            mob.next_to(cheetah, DOWN, buff=0.1)

        self.add(vel_arrow, accel_label)
        cheetah.add_updater(update_cheetah)
        vel_arrow.add_updater(update_vel_arrow)
        accel_label.add_updater(update_accel_label)
        
        # Run movement animation
        self.play(t_tracker.animate.set_value(1.0), run_time=3, rate_func=linear)
        
        # Remove updaters to prevent further processing
        cheetah.clear_updaters()
        vel_arrow.clear_updaters()
        accel_label.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: Display text 'Rate of Change' and link it to the acceleration of the [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/cheetah.svg]. [Color: #FFFFFF]
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        # Conceptual labels
        roc_text = Text("Rate of Change", font_size=22, color="#FFFFFF")
        self.place_at_grid(roc_text, "B3", scale_factor=1.0)
        
        formula = MathTex(r"\frac{dv}{dt} = a", color="#FFFFFF")
        self.place_at_grid(formula, "C3", scale_factor=1.2)
        
        # Connections
        link_1 = Arrow(roc_text.get_bottom(), formula.get_top(), color=WHITE, buff=0.1)
        link_2 = Arrow(formula.get_bottom(), accel_label.get_top(), color=WHITE, buff=0.2)
        
        self.play(Write(roc_text))
        self.play(Create(link_1), FadeIn(formula))
        self.play(Create(link_2))
        
        self.wait(3)
