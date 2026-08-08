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

class Section5Scene(TeachingScene):
    def construct(self):
        title = "Connecting the Dots to Pi"
        lecture_lines = [
            "Total collisions depend on the arc's size.",
            "We are fitting small steps into a half-circle.",
            "As the large mass increases by powers of 100...",
            "The number of steps precisely matches digits of Pi.",
            "Geometry translates physics directly into this famous constant."
        ]
        self.setup_layout(title, lecture_lines)
        
        cyan_color = "#00FFFF"
        orange_color = "#FFA500"

        # === Animation for Lecture Line 1 ===
        # Total collisions depend on the arc's size.
        self.lecture[0].set_color(YELLOW)
        
        # Circle in the center (scaled coordinates phase space)
        circle = Circle(radius=2.0, color=cyan_color)
        self.place_in_area(circle, 'B2', 'E5')
        circle_center = circle.get_center()
        self.play(Create(circle))
        
        # Angle theta for the arc. For k=1 (100:1), N=31. 
        # Each step is roughly theta = pi / 31.4...
        theta_val = PI / 10 # Slightly larger for visibility
        
        # Create a group for arcs
        arcs = VGroup()
        
        # Draw the first few arcs
        for i in range(3):
            arc = Arc(radius=2.0, start_angle=PI - (i+1)*theta_val, angle=theta_val, color=orange_color, stroke_width=6)
            arc.move_to(circle_center)
            arcs.add(arc)
            self.play(Create(arc), run_time=0.5)
            
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We are fitting small steps into a half-circle.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Add more arcs to fill the top half
        remaining_steps = 7 # Total ~10 steps for this demo theta
        for i in range(3, remaining_steps):
            arc = Arc(radius=2.0, start_angle=PI - (i+1)*theta_val, angle=theta_val, color=orange_color, stroke_width=6)
            arc.move_to(circle_center)
            arcs.add(arc)
            self.play(Create(arc), run_time=0.2)

        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # As the large mass increases by powers of 100...
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Labels for mass ratio and collision count
        mass_label = MathTex("M/m = 100^1", color=WHITE)
        self.place_at_grid(mass_label, 'B6', scale_factor=0.8)
        
        collision_label_prefix = MathTex("N = ", color=WHITE)
        self.place_at_grid(collision_label_prefix, 'D6', scale_factor=0.8)
        collision_label_prefix.shift(LEFT * 0.4)
        
        count_tracker = ValueTracker(31)
        counter_val = DecimalNumber(31, num_decimal_places=0, color=orange_color)
        counter_val.scale(0.8)
        counter_val.next_to(collision_label_prefix, RIGHT, buff=0.1)
        counter_val.add_updater(lambda d: d.set_value(count_tracker.get_value()))
        
        self.play(Write(mass_label), Write(collision_label_prefix), Write(counter_val))
        
        # Change k from 1 to 2
        new_mass_label = MathTex("M/m = 100^2", color=WHITE)
        self.place_at_grid(new_mass_label, 'B6', scale_factor=0.8)
        
        self.play(
            Transform(mass_label, new_mass_label),
            count_tracker.animate.set_value(314),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The number of steps precisely matches digits of Pi.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Update mass label to show higher power
        final_mass_label = MathTex("M/m = 100^5", color=WHITE)
        self.place_at_grid(final_mass_label, 'B6', scale_factor=0.8)
        
        # Visual representation of "many tiny arcs"
        dense_arc = Arc(radius=2.0, start_angle=0, angle=PI, color=orange_color, stroke_width=2)
        dense_arc.move_to(circle_center)
        
        target_count = 314159
        
        self.play(
            Transform(mass_label, final_mass_label),
            FadeOut(arcs),
            Create(dense_arc),
            count_tracker.animate.set_value(target_count),
            run_time=3,
            rate_func=linear
        )
        
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Geometry translates physics directly into this famous constant.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        pi_formula = MathTex(r"N \approx \pi \cdot \sqrt{M/m}", color=YELLOW)
        self.place_at_grid(pi_formula, 'E4', scale_factor=0.8)
        # Shift it a bit to the right to not overlap circle
        pi_formula.shift(RIGHT * 0.5)
        
        self.play(Write(pi_formula))
        
        self.wait(2)
