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

class Section6Scene(TeachingScene):
    def construct(self):
        title = "Application and Wrap-up: The Steering Wheel"
        lines = [
            "Derivatives help us navigate a changing world.",
            "They guide self-driving cars along smooth paths.",
            "Calculus turns complex motion into predictable patterns."
        ]
        self.setup_layout(title, lines)

        # Colors
        ROAD_COLOR = GREY_B
        TANGENT_COLOR = "#FFD700"
        WHEEL_COLOR = "#58C4DD"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Winding road - adjusted to fit area B3-F6 (Width ~3)
        road_func = lambda t: np.array([t - 1.5, 0.7 * np.sin(1.5 * t) + 0.3 * np.cos(3 * t), 0])
        road = ParametricFunction(road_func, t_range=[0, 3], color=ROAD_COLOR)
        # Fix for Issue 36: Move road from A3 to B3
        self.place_in_area(road, 'B3', 'F6', scale_factor=1.0)
        
        road_boundary = DashedVMobject(road, num_dashes=40)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png
        # Fix for Issue 21: Use ImageMobject
        try:
            car = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
            car.height = 0.5
        except Exception:
            # Fallback if asset is missing
            car = Triangle(color=WHITE, fill_opacity=1).scale(0.2).rotate(-PI/2)
            
        t_tracker = ValueTracker(0)
        
        # Track angle for smooth rotation
        car.current_angle = 0
        
        def update_car(m):
            t = t_tracker.get_value()
            prop = t / 3
            pos = road.point_from_proportion(prop)
            
            # Use small delta for tangent
            epsilon = 0.01
            p1 = road.point_from_proportion(max(0, prop - epsilon))
            p2 = road.point_from_proportion(min(1, prop + epsilon))
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            
            m.move_to(pos)
            m.rotate(angle - m.current_angle)
            m.current_angle = angle

        # Initialize car position and angle
        update_car(car) 
        car.add_updater(update_car)
        
        self.play(Create(road_boundary), run_time=1.5)
        self.add(car)
        self.play(t_tracker.animate.set_value(3), run_time=4, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(TANGENT_COLOR)
        
        # Tangent arrow
        tangent_arrow = Arrow(start=LEFT, end=RIGHT, color=TANGENT_COLOR, buff=0)
        tangent_arrow.scale(0.6)
        tangent_arrow.current_angle = 0

        def update_arrow(m):
            t = t_tracker.get_value()
            prop = t / 3
            pos = road.point_from_proportion(prop)
            
            epsilon = 0.01
            p1 = road.point_from_proportion(max(0, prop - epsilon))
            p2 = road.point_from_proportion(min(1, prop + epsilon))
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            
            m.move_to(pos)
            m.rotate(angle - m.current_angle)
            m.current_angle = angle

        # Initialize arrow
        t_tracker.set_value(0)
        update_arrow(tangent_arrow)
        tangent_arrow.add_updater(update_arrow)
        
        self.play(Create(tangent_arrow))
        self.play(t_tracker.animate.set_value(3), run_time=4, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHEEL_COLOR)
        
        # Steering Wheel
        wheel_circle = Circle(radius=0.5, color=WHEEL_COLOR)
        wheel_cross1 = Line(start=LEFT*0.5, end=RIGHT*0.5, color=WHEEL_COLOR)
        wheel_cross2 = Line(start=UP*0.5, end=DOWN*0.5, color=WHEEL_COLOR)
        wheel = VGroup(wheel_circle, wheel_cross1, wheel_cross2)
        
        # Fix for Issue 35: Move wheel to B2 and scale 0.9
        self.place_at_grid(wheel, 'B2', scale_factor=0.9)
        wheel.current_angle = 0

        def update_wheel(m):
            t = t_tracker.get_value()
            prop = t / 3
            epsilon = 0.01
            p1 = road.point_from_proportion(max(0, prop - epsilon))
            p2 = road.point_from_proportion(min(1, prop + epsilon))
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            
            m.rotate(angle - m.current_angle)
            m.current_angle = angle

        # Initialize wheel
        t_tracker.set_value(0)
        update_wheel(wheel)
        wheel.add_updater(update_wheel)
        
        self.play(FadeIn(wheel))
        self.play(t_tracker.animate.set_value(3), run_time=6, rate_func=linear)
        
        self.wait(2)
