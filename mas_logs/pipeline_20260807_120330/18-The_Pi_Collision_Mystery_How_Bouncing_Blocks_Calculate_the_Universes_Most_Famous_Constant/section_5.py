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
        highlight_color = YELLOW

        # === Animation for Lecture Line 1 ===
        # Total collisions depend on the arc's size.
        self.lecture[0].set_color(highlight_color)
        
        # Circle in the center (scaled coordinates phase space)
        # Using B2 to E5 to center the circle in the grid area
        circle = Circle(radius=1.8, color=cyan_color)
        self.place_in_area(circle, 'B2', 'E5')
        circle_center = circle.get_center()
        self.play(Create(circle))
        
        # Angle theta for the arc.
        theta_val = PI / 8 
        arcs = VGroup()
        
        # Create first 4 arcs (filling roughly half of the upper semi-circle)
        initial_arcs = VGroup()
        for i in range(4):
            arc = Arc(radius=1.8, start_angle=PI - (i+1)*theta_val, angle=theta_val, color=orange_color, stroke_width=6)
            arc.move_to(circle_center)
            initial_arcs.add(arc)
        
        self.play(AnimationGroup(*[Create(a) for a in initial_arcs], lag_ratio=0.5), run_time=1.5)
        arcs.add(initial_arcs)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We are fitting small steps into a half-circle.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(highlight_color)
        
        # Add 4 more arcs to complete the top half-circle
        more_arcs = VGroup()
        for i in range(4, 8):
            arc = Arc(radius=1.8, start_angle=PI - (i+1)*theta_val, angle=theta_val, color=orange_color, stroke_width=6)
            arc.move_to(circle_center)
            more_arcs.add(arc)
            
        self.play(AnimationGroup(*[Create(a) for a in more_arcs], lag_ratio=0.3), run_time=1.2)
        arcs.add(more_arcs)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # As the large mass increases by powers of 100...
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(highlight_color)
        
        # Asset: Mass icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/mass.svg]
        # Issue 40 fix: Move mass_icon to A6 to avoid overlap with arc trajectory
        mass_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mass.svg", color=WHITE)
        self.place_at_grid(mass_icon, 'A6', scale_factor=0.5)
        
        mass_label = MathTex("M/m = 100^1", color=WHITE)
        mass_label.scale(0.7)
        mass_label.next_to(mass_icon, DOWN, buff=0.1)
        
        # Issue 41 fix: Move collision label to C5-C6 area to avoid obstruction
        collision_label_prefix = MathTex("N = ", color=WHITE)
        self.place_in_area(collision_label_prefix, 'C5', 'C6', scale_factor=0.8)
        collision_label_prefix.shift(LEFT * 0.4)
        
        count_tracker = ValueTracker(31)
        counter_val = DecimalNumber(31, num_decimal_places=0, color=orange_color)
        counter_val.scale(0.8)
        counter_val.next_to(collision_label_prefix, RIGHT, buff=0.1)
        counter_val.add_updater(lambda d: d.set_value(count_tracker.get_value()))
        
        self.play(
            FadeIn(mass_icon), 
            FadeIn(mass_label), 
            FadeIn(collision_label_prefix), 
            FadeIn(counter_val)
        )
        
        # Show mass ratio increasing
        new_mass_label = MathTex("M/m = 100^2", color=WHITE)
        new_mass_label.scale(0.7)
        new_mass_label.move_to(mass_label.get_center())
        
        self.play(
            Transform(mass_label, new_mass_label),
            count_tracker.animate.set_value(314),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The number of steps precisely matches digits of Pi.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(highlight_color)
        
        # Update mass label to show higher power
        final_mass_label = MathTex("M/m = 100^5", color=WHITE)
        final_mass_label.scale(0.7)
        final_mass_label.move_to(mass_label.get_center())
        
        # Visual representation of "many tiny arcs" filling the semi-circle
        dense_arc = Arc(radius=1.8, start_angle=0, angle=PI, color=orange_color, stroke_width=2)
        dense_arc.move_to(circle_center)
        # Position it on the top half
        dense_arc.rotate(PI, about_point=circle_center)
        
        target_count = 314159
        
        # Transition to tiny arcs and high counter
        self.play(
            Transform(mass_label, final_mass_label),
            FadeOut(arcs),
            Create(dense_arc),
            count_tracker.animate.set_value(target_count),
            run_time=2,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Geometry translates physics directly into this famous constant.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(highlight_color)
        
        # Issue 42 fix: Move formula to F4-F6 area to avoid crowded layout
        pi_formula = MathTex(r"N \approx \pi \sqrt{M/m}", color=highlight_color)
        self.place_in_area(pi_formula, 'F4', 'F6', scale_factor=0.7)
        
        self.play(Write(pi_formula))
        self.wait(2)
