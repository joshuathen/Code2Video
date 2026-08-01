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
        # Initial velocity vector (very small)
        vel_arrow = Arrow(LEFT * 0.1, RIGHT * 0.1, color="#00FF00", buff=0, stroke_width=4)
        vel_arrow.next_to(cheetah, UP, buff=0.1)
        
        accel_label = Text("a = constant", font_size=16, color="#00FF00")
        accel_label.next_to(cheetah, DOWN, buff=0.1)
        
        self.play(FadeIn(vel_arrow), FadeIn(accel_label))
        
        # Movement: Accelerated motion from F1 to F6
        # Use ease_in_quad to simulate constant acceleration (x ~ t^2)
        target_pos = self.grid["F6"]
        
        # We define a custom updater for the arrow to follow and scale, 
        # but keep it simple to ensure fast rendering.
        def update_indicators(mob):
            # Move with cheetah
            vel_arrow.next_to(cheetah, UP, buff=0.1)
            accel_label.next_to(cheetah, DOWN, buff=0.1)
            # Scale arrow width based on progress (linear growth of velocity)
            # Using cheetah's position relative to start as a proxy for time
            progress = (cheetah.get_x() - self.grid["F1"][0]) / (self.grid["F6"][0] - self.grid["F1"][0])
            new_width = 0.2 + 1.2 * (progress**0.5) # simple growth
            vel_arrow.set_width(new_width, stretch=True, about_edge=LEFT)

        vel_arrow.add_updater(update_indicators)

        self.play(
            cheetah.animate.move_to(target_pos),
            run_time=3,
            rate_func=rate_functions.ease_in_quad
        )
        
        vel_arrow.remove_updater(update_indicators)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: Display text 'Rate of Change' and link it to the acceleration of the [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/cheetah.svg]. [Color: #FFFFFF]
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        # Conceptual labels
        roc_text = Text("Rate of Change", font_size=22, color="#FFFFFF")
        self.place_at_grid(roc_text, "B3")
        
        formula = MathTex(r"\frac{dv}{dt} = a", color="#FFFFFF")
        self.place_at_grid(formula, "C3", scale_factor=1.2)
        
        # Connections
        link_1 = Arrow(roc_text.get_bottom(), formula.get_top(), color=WHITE, buff=0.1)
        # Link formula to the acceleration label currently at F6
        link_2 = Arrow(formula.get_bottom(), accel_label.get_top(), color=WHITE, buff=0.1)
        
        self.play(Write(roc_text))
        self.play(Create(link_1), FadeIn(formula))
        self.play(Create(link_2))
        
        self.wait(3)