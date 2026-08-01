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
        self.setup_layout("Summary and Real-World Application", [
            "Vectors are powerful tools for direction and magnitude.",
            "They drive physics in video games and navigation.",
            "Master vectors to understand how the world moves."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Color transition for lecture line 1
        self.play(self.lecture[0].animate.set_color(WHITE))

        # Car icon (triangle)
        car = Triangle(color=BLUE, fill_opacity=1).rotate(-PI/2)
        # Issue 36: Move car from D1 to D2, scale to 0.4
        self.place_at_grid(car, "D2", scale_factor=0.4)
        
        # Velocity vector (white)
        velocity_vector = Arrow(
            start=LEFT * 0.5, end=RIGHT * 0.5, 
            color="#FFFFFF", buff=0, stroke_width=4
        )
        velocity_vector.next_to(car, RIGHT, buff=0)
        
        # Acceleration vector (yellow)
        acceleration_vector = Arrow(
            start=LEFT * 0.3, end=RIGHT * 0.3, 
            color="#FFFF00", buff=0, stroke_width=4
        )
        acceleration_vector.next_to(car, UP, buff=0.1)

        # Labels for vectors
        vel_label = Text("Velocity", font_size=14, color="#FFFFFF")
        accel_label = Text("Accel", font_size=14, color="#FFFF00")
        vel_label.next_to(velocity_vector, RIGHT, buff=0.1)
        accel_label.next_to(acceleration_vector, UP, buff=0.1)

        # Group components to move together
        car_group = VGroup(car, velocity_vector, acceleration_vector, vel_label, accel_label)
        
        self.play(FadeIn(car_group))
        self.play(car_group.animate.move_to(self.grid["D4"]), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color transition for lecture line 2
        self.play(self.lecture[1].animate.set_color("#FFFF00"))

        # Change direction and size of velocity vector
        new_vel_vec = Arrow(
            start=car.get_center(), end=car.get_center() + UP * 1.5 + RIGHT * 0.5,
            color="#FFFFFF", buff=0, stroke_width=4
        )
        
        # Perform rotation and scale change
        self.play(
            car.animate.rotate(PI/4),
            velocity_vector.animate.become(new_vel_vec),
            vel_label.animate.next_to(new_vel_vec.get_end(), RIGHT, buff=0.1),
            acceleration_vector.animate.rotate(PI/4).shift(UP*0.2),
            accel_label.animate.shift(UP*0.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color transition for lecture line 3
        self.play(self.lecture[2].animate.set_color("#FFC0CB")) # Light Pink

        # Fade out car elements
        self.play(FadeOut(car_group))

        # Collage icons
        # 1. Satellite (Circle + Rectangles)
        satellite = VGroup(
            Circle(radius=0.2, color=BLUE, fill_opacity=0.5),
            Rectangle(width=0.4, height=0.1, color=GRAY, fill_opacity=1).shift(LEFT*0.25),
            Rectangle(width=0.4, height=0.1, color=GRAY, fill_opacity=1).shift(RIGHT*0.25)
        )
        sat_label = Text("GPS", font_size=18, color=BLUE)
        self.place_at_grid(satellite, "B2", scale_factor=0.8)
        sat_label.next_to(satellite, UP, buff=0.2)

        # 2. Game Controller (Rounded Rect + 2 circles)
        controller = VGroup(
            RoundedRectangle(corner_radius=0.1, height=0.3, width=0.5, color=GREY_B, fill_opacity=1),
            Circle(radius=0.05, color=BLACK, fill_opacity=1).shift(LEFT*0.1),
            Circle(radius=0.05, color=BLACK, fill_opacity=1).shift(RIGHT*0.1)
        )
        game_label = Text("Games", font_size=18, color=GREY_B)
        # Issue 37: Move controller to E5
        self.place_at_grid(controller, "E5", scale_factor=0.8)
        game_label.next_to(controller, DOWN, buff=0.2)

        # 3. Map (Square with path)
        map_icon = VGroup(
            Square(side_length=0.4, color=GREEN_E, fill_opacity=0.3),
            Line(LEFT*0.15 + DOWN*0.15, RIGHT*0.15 + UP*0.15, color=RED)
        )
        map_label = Text("Map", font_size=18, color=GREEN_E)
        self.place_at_grid(map_icon, "B5", scale_factor=0.8)
        map_label.next_to(map_icon, UP, buff=0.2)

        # Central spinning vector
        center_vector = Arrow(ORIGIN, RIGHT * 1.0, color="#FFC0CB", buff=0)
        # Issue 38: scale_factor=0.7
        self.place_at_grid(center_vector, "D4", scale_factor=0.7)
        
        # Collage appearance
        self.play(
            FadeIn(satellite), FadeIn(sat_label),
            FadeIn(controller), FadeIn(game_label),
            FadeIn(map_icon), FadeIn(map_label),
            GrowArrow(center_vector)
        )
        
        # Central vector spins
        self.play(Rotate(center_vector, angle=2*PI, about_point=center_vector.get_start()), run_time=3, rate_func=linear)
        self.wait(2)
