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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initialize Scene with correct lecture lines
        lecture_lines = [
            "Let's see how light moves from air into water.",
            "An incoming ray hits the surface at an angle.",
            "It bends toward the normal in the denser water.",
            "Snell's Law mathematically predicts this exact path change.",
            "Changing the entry angle shifts the refraction accordingly."
        ]
        self.setup_layout("Snell's Law: The Mathematical Relationship", lecture_lines)

        # Constants
        n1 = 1.00  # Air
        n2 = 1.33  # Water
        
        # Grid positions
        p_intersection = self.grid['D3']
        normal_top = self.grid['B3']
        normal_bottom = self.grid['F3']
        boundary_start = self.grid['D1']
        boundary_end = self.grid['D6']

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Boundary
        boundary = Line(boundary_start, boundary_end, color=BLUE_B, stroke_width=4)
        
        # Labels and Icons (Assets)
        air_label = Text("Air (n=1.0)", font_size=20, color=WHITE)
        water_label = Text("Water (n=1.33)", font_size=20, color=WHITE)
        
        # Using Issue 38/39 constraints
        self.place_in_area(air_label, 'C3', 'C5', scale_factor=0.8)
        self.place_in_area(water_label, 'E3', 'E5', scale_factor=0.8)
        
        # Load Assets (Issue 27)
        air_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/air.svg")
        water_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/water.svg")
        self.place_at_grid(air_icon, "C2", scale_factor=0.4)
        self.place_at_grid(water_icon, "E2", scale_factor=0.4)
        
        self.play(Create(boundary), FadeIn(air_icon), Write(air_label), FadeIn(water_icon), Write(water_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00") # Green
        
        # Normal line
        normal_line = DashedLine(normal_top, normal_bottom, color=GRAY)
        
        # Value Tracker for theta1 (angle from normal)
        angle_tracker = ValueTracker(45 * DEGREES)
        
        # Incident Ray (Green)
        incident_ray = Line(color="#00FF00", stroke_width=6)
        
        def update_incident(ray):
            t1 = angle_tracker.get_value()
            # Ray direction vector (coming from top-left): [sin(t1), -cos(t1), 0]
            # Ray start is 2 units away from intersection along that vector reversed
            start = p_intersection + 2 * np.array([-np.sin(t1), np.cos(t1), 0])
            ray.set_points_as_corners([start, p_intersection])

        incident_ray.add_updater(update_incident)
        
        # Incidence Arc
        arc_incident = Arc(radius=0.5, start_angle=90*DEGREES, color="#00FF00")
        arc_incident.add_updater(lambda m: m.set_angle(-angle_tracker.get_value()).move_arc_center_to(p_intersection))
        
        theta1_text = Text("θ₁", font_size=20, color="#00FF00")
        theta1_text.add_updater(lambda m: m.move_to(p_intersection + 0.8 * np.array([-np.sin(angle_tracker.get_value()/2), np.cos(angle_tracker.get_value()/2), 0])))

        self.play(Create(normal_line), Create(incident_ray), Create(arc_incident), Write(theta1_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00") # Yellow
        
        # Refracted Ray (Yellow)
        refracted_ray = Line(color="#FFFF00", stroke_width=6)
        
        def update_refracted(ray):
            t1 = angle_tracker.get_value()
            t2 = np.arcsin((n1/n2) * np.sin(t1))
            # Ray direction vector: [sin(t2), -cos(t2), 0]
            end = p_intersection + 2 * np.array([np.sin(t2), -np.cos(t2), 0])
            ray.set_points_as_corners([p_intersection, end])
            
        refracted_ray.add_updater(update_refracted)
        
        # Refraction Arc
        arc_refracted = Arc(radius=0.5, start_angle=270*DEGREES, color="#FFFF00")
        arc_refracted.add_updater(lambda m: m.set_angle(np.arcsin((n1/n2) * np.sin(angle_tracker.get_value()))).move_arc_center_to(p_intersection))
        
        theta2_text = Text("θ₂", font_size=20, color="#FFFF00")
        theta2_text.add_updater(lambda m: m.move_to(p_intersection + 0.8 * np.array([np.sin(np.arcsin((n1/n2) * np.sin(angle_tracker.get_value()))/2), -np.cos(np.arcsin((n1/n2) * np.sin(angle_tracker.get_value()))/2), 0])))

        self.play(Create(refracted_ray), Create(arc_refracted), Write(theta2_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        
        # Formula (Issue 40 position)
        formula = Text("n₁ sin(θ₁) = n₂ sin(θ₂)", font_size=24, color=WHITE)
        self.place_in_area(formula, 'A3', 'B6', scale_factor=0.9)
        
        self.play(Write(formula))
        self.play(formula.animate.set_color(YELLOW).scale(1.1), run_time=0.5)
        self.play(formula.animate.set_color(WHITE).scale(1/1.1), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        
        # Sweep angle to show dynamics
        self.play(angle_tracker.animate.set_value(65 * DEGREES), run_time=2, rate_func=linear)
        self.wait(0.5)
        self.play(angle_tracker.animate.set_value(25 * DEGREES), run_time=2, rate_func=linear)
        self.wait(0.5)
        self.play(angle_tracker.animate.set_value(45 * DEGREES), run_time=1, rate_func=linear)
        
        self.wait(2)
