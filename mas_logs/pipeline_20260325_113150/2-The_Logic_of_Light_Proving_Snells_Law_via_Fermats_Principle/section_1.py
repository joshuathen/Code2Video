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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        self.setup_layout("The Lifeguard's Dilemma (Introduction)", [
            "A lifeguard must reach a swimmer as fast as possible.",
            "The direct straight path isn't always the fastest route.",
            "Running on sand is faster than swimming through water.",
            "The bent path arrives first by maximizing speed on land.",
            "Light also follows this Principle of Least Time."
        ])

        # Positions based on grid coordinates
        pos_A = self.grid["B1"]
        pos_B = self.grid["E6"]
        
        # Interface is the boundary between beach and water
        interface_y = -0.3
        pos_interface_straight = np.array([3.0, interface_y, 0])
        pos_interface_bent = np.array([4.0, interface_y, 0])

        # === Animation for Lecture Line 1 ===
        # Establish the environment: replaced ImageMobjects with Rectangles to avoid missing asset errors
        beach = Rectangle(width=5, height=3, fill_color="#C2B280", fill_opacity=1, stroke_width=0)
        self.place_in_area(beach, "A1", "C6")
        
        water = Rectangle(width=5, height=3, fill_color="#0077BE", fill_opacity=1, stroke_width=0)
        self.place_in_area(water, "D1", "F6")

        # Replaced image assets with Vector Groups
        lifeguard_asset = VGroup(
            Circle(radius=0.3, color=RED, fill_opacity=1),
            Text("L", font_size=20, color=WHITE)
        )
        self.place_at_grid(lifeguard_asset, "B1", scale_factor=0.6)
        
        swimmer_asset = VGroup(
            Circle(radius=0.3, color=BLUE_E, fill_opacity=1),
            Text("S", font_size=20, color=WHITE)
        )
        self.place_at_grid(swimmer_asset, "E6", scale_factor=0.6)
        
        # Labels for start/end points
        label_A = Text("A", font_size=24, color=WHITE)
        self.place_at_grid(label_A, "A2", scale_factor=0.8)
        
        label_B = Text("B", font_size=24, color=WHITE)
        self.place_at_grid(label_B, "F5", scale_factor=0.8)

        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(FadeIn(beach), FadeIn(water))
        self.play(FadeIn(lifeguard_asset), FadeIn(swimmer_asset), Write(label_A), Write(label_B))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#888888"))
        
        path_straight_sand = Line(pos_A, pos_interface_straight, color="#888888")
        path_straight_water = Line(pos_interface_straight, pos_B, color="#888888")
        path_straight = VGroup(path_straight_sand, path_straight_water)
        
        dot_straight = Dot(pos_A, color="#888888")
        
        self.play(Create(path_straight))
        
        # Run straight path race
        self.play(
            Succession(
                MoveAlongPath(dot_straight, path_straight_sand, rate_func=linear, run_time=1.45),
                MoveAlongPath(dot_straight, path_straight_water, rate_func=linear, run_time=2.91)
            )
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Display the bent path that stays longer on land
        path_bent_sand = Line(pos_A, pos_interface_bent, color="#FFFF00")
        path_bent_water = Line(pos_interface_bent, pos_B, color="#FFFF00")
        path_bent = VGroup(path_bent_sand, path_bent_water)
        
        dot_bent = Dot(pos_A, color="#FFFF00")
        
        self.play(Create(path_bent))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        
        # Reset dots for a comparison race
        self.play(
            dot_straight.animate.move_to(pos_A),
            dot_bent.animate.move_to(pos_A),
            run_time=0.5
        )
        
        # Comparative Race
        self.play(
            Succession(
                MoveAlongPath(dot_straight, path_straight_sand, rate_func=linear, run_time=1.45),
                MoveAlongPath(dot_straight, path_straight_water, rate_func=linear, run_time=2.91)
            ),
            Succession(
                MoveAlongPath(dot_bent, path_bent_sand, rate_func=linear, run_time=1.9),
                MoveAlongPath(dot_bent, path_bent_water, rate_func=linear, run_time=2.12)
            )
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFD700"))
        
        least_time_label = Text("Least Time", font_size=24, color="#FFD700")
        self.place_at_grid(least_time_label, "B6", scale_factor=0.8)
        
        self.play(
            path_bent.animate.set_stroke(width=8),
            Write(least_time_label)
        )
        self.wait(2)
