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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initialize layout
        lecture_lines = [
            "The constant e represents the spirit of continuous growth.",
            "Usually, e pushes us along the real number line.",
            "But what if our growth rate is purely imaginary?"
        ]
        self.setup_layout("The Growth Engine: What is 'e'?", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#90EE90")
        
        # Create 'e' text and engine icon
        e_text = Text("e", color="#90EE90").scale(3)
        engine_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/engine.svg").scale(0.8)
        engine_icon.set_color(WHITE)
        
        # Group and place
        growth_engine = VGroup(e_text, engine_icon).arrange(RIGHT, buff=0.5)
        self.place_in_area(growth_engine, "B2", "C5")
        
        self.play(FadeIn(growth_engine))
        
        # Pulsing effect
        self.play(
            growth_engine.animate.scale(1.15),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700") # Yellow/Gold for "real" growth
        
        # Create Number Line
        num_line = NumberLine(
            x_range=[0, 5, 1],
            length=4.5,
            include_numbers=True,
            label_constructor=Text,
            include_tip=True,
            color=WHITE
        )
        self.place_in_area(num_line, "E1", "E6")
        
        # Point and Growth Arrow
        # Starting point at 1
        current_val = ValueTracker(1)
        dot = Dot(num_line.n2p(1), color="#FFD700")
        
        # Velocity/Growth arrow
        growth_arrow = Arrow(
            start=LEFT * 0.5, 
            end=RIGHT * 0.5, 
            color="#FFD700", 
            buff=0,
            stroke_width=5
        )
        growth_arrow.next_to(dot, RIGHT, buff=0.1)

        # Updaters for movement
        def update_dot(mob):
            mob.move_to(num_line.n2p(current_val.get_value()))
            
        def update_arrow(mob):
            mob.next_to(dot, RIGHT, buff=0.1)

        dot.add_updater(update_dot)
        growth_arrow.add_updater(update_arrow)

        self.play(Create(num_line), FadeIn(dot), GrowArrow(growth_arrow))
        
        # Animate growth from 1 to ~4.5
        # We simulate "e^t" by tracking value
        self.play(current_val.animate.set_value(4.2), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#ADD8E6")
        
        # Stop updaters before rotation
        dot.remove_updater(update_dot)
        growth_arrow.remove_updater(update_arrow)

        # Rotate arrow to point upwards (Imaginary direction)
        # and change color to light blue
        new_arrow_color = "#ADD8E6"
        
        self.play(
            Rotate(growth_arrow, angle=PI/2, about_point=dot.get_center()),
            growth_arrow.animate.set_color(new_arrow_color),
            dot.animate.set_color(new_arrow_color),
            run_time=2
        )
        
        # Final pulsing to emphasize the change
        self.play(
            growth_arrow.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=1
        )
        
        self.wait(2)
