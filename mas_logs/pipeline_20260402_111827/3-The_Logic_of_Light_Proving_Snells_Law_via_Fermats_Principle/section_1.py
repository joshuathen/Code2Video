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
        title = "The Refraction Mystery"
        lines = [
            "Why does light bend when moving between different materials?",
            "Light travels at speed v1 here and v2 there.",
            "Imagine a lifeguard reaching a swimmer in the ocean.",
            "Running is faster than swimming, so paths aren't straight.",
            "Light also takes a path to minimize travel time."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create Glass - spanning the full grid area A1-F6
        glass_outline = Rectangle(width=5, height=5, color=WHITE, stroke_width=2)
        self.place_in_area(glass_outline, "A1", "F6")
        
        # Water Level - spanning the bottom half D1-F6
        water_rect = Rectangle(width=5, height=2.5, fill_color="#1E90FF", fill_opacity=0.6, stroke_width=0)
        self.place_in_area(water_rect, "D1", "F6")
        
        # Broken Pencil
        pencil_top = Line(start=self.grid["B3"] + LEFT*0.5, end=self.grid["C4"] + RIGHT*0.2, color="#D4AF37", stroke_width=6)
        pencil_bottom = Line(start=self.grid["C4"] + RIGHT*0.2, end=self.grid["E5"] + RIGHT*0.8, color="#D4AF37", stroke_width=6)
        pencil_bottom.shift(RIGHT*0.3 + DOWN*0.1) # Simulate offset/refraction
        
        self.play(Create(glass_outline), FadeIn(water_rect))
        self.play(Create(pencil_top), Create(pencil_bottom))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        self.play(FadeOut(glass_outline), FadeOut(water_rect), FadeOut(pencil_top), FadeOut(pencil_bottom))
        
        # Interface Line
        interface_y = (self.grid["C3"][1] + self.grid["D3"][1]) / 2
        interface_line = Line(start=[0.5, interface_y, 0], end=[5.5, interface_y, 0], color=WHITE)
        
        # Sand and Water Regions
        sand_bg = Rectangle(width=5, height=3, fill_color="#EDC9AF", fill_opacity=0.4, stroke_width=0)
        self.place_in_area(sand_bg, "A1", "C6")
        sand_bg.shift(UP * 0.15) # Adjust to align with interface_y
        
        water_bg = Rectangle(width=5, height=3, fill_color="#1E90FF", fill_opacity=0.4, stroke_width=0)
        self.place_in_area(water_bg, "D1", "F6")
        water_bg.shift(DOWN * 0.15)
        
        self.play(Create(interface_line), FadeIn(sand_bg), FadeIn(water_bg))
        
        v1_label = Text("v₁", color="#00FF00")
        self.place_at_grid(v1_label, "B3", scale_factor=0.8)
        v2_label = Text("v₂", color="#FF0000")
        self.place_at_grid(v2_label, "E3", scale_factor=0.8)
        
        self.play(Write(v1_label), Write(v2_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        pointA = Dot(self.grid["B1"], color=WHITE)
        labelA = Text("Lifeguard (A)", font_size=16).next_to(pointA, UP)
        pointB = Dot(self.grid["E6"], color=WHITE)
        labelB = Text("Swimmer (B)", font_size=16).next_to(pointB, DOWN)
        
        self.play(Create(pointA), Write(labelA), Create(pointB), Write(labelB))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Dashed Straight Line
        straight_path = DashedLine(pointA.get_center(), pointB.get_center(), color=GRAY)
        self.play(Create(straight_path))
        
        # Traveling Dot
        traveler = Dot(color=WHITE, radius=0.08)
        
        # Calculate intersection with interface for timing mathematically
        # instead of point_from_proportion which fails on DashedLine
        # Intersection of line segment A-B with horizontal line y=interface_y
        # y = y1 + t(y2-y1) -> interface_y = y1 + t(y2-y1) -> t = (interface_y - y1) / (y2 - y1)
        y1, y2 = pointA.get_center()[1], pointB.get_center()[1]
        t = (interface_y - y1) / (y2 - y1)
        intersection_point = pointA.get_center() + t * (pointB.get_center() - pointA.get_center())
        
        # Animation
        self.play(MoveAlongPath(traveler, Line(pointA.get_center(), intersection_point)), run_time=1, rate_func=linear)
        self.play(MoveAlongPath(traveler, Line(intersection_point, pointB.get_center())), run_time=2, rate_func=linear)
        self.play(FadeOut(traveler))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Normal line at an optimal refraction point (shifted right relative to straight path)
        optimal_x_pos_val = intersection_point[0] + 1.0
        optimal_point = np.array([optimal_x_pos_val, interface_y, 0])
        normal_line = DashedLine(optimal_point + UP*1.5, optimal_point + DOWN*1.5, color="#808080")
        
        path_segments = VGroup(
            Line(pointA.get_center(), optimal_point, color=WHITE),
            Line(optimal_point, pointB.get_center(), color=WHITE)
        )
        
        angle1 = Arc(radius=0.5, start_angle=PI/2, angle=-PI/4, arc_center=optimal_point)
        theta1 = Text("θ₁", font_size=20).next_to(angle1, UP+RIGHT, buff=0.1)
        angle2 = Arc(radius=0.5, start_angle=-PI/2, angle=PI/6, arc_center=optimal_point)
        theta2 = Text("θ₂", font_size=20).next_to(angle2, DOWN+LEFT, buff=0.1)
        
        v1_vec = Arrow(start=pointA.get_center(), end=pointA.get_center() + RIGHT*0.8, color="#00FF00", buff=0)
        v2_vec = Arrow(start=pointB.get_center(), end=pointB.get_center() + RIGHT*0.4, color="#FF0000", buff=0)
        
        self.play(Create(normal_line), Create(path_segments))
        self.play(Create(angle1), Write(theta1), Create(angle2), Write(theta2))
        self.play(GrowArrow(v1_vec), GrowArrow(v2_vec))
        
        pointX = Dot(optimal_point, color=WHITE)
        self.add(pointX)
        
        x_tracker = ValueTracker(optimal_x_pos_val)
        
        def update_path(path):
            curr_x = x_tracker.get_value()
            new_p = np.array([curr_x, interface_y, 0])
            path[0].set_points_as_corners([pointA.get_center(), new_p])
            path[1].set_points_as_corners([new_p, pointB.get_center()])
            pointX.move_to(new_p)

        path_segments.add_updater(update_path)
        
        self.play(x_tracker.animate.set_value(optimal_x_pos_val - 1.5), run_time=1)
        self.play(x_tracker.animate.set_value(optimal_x_pos_val + 0.5), run_time=1)
        path_segments.clear_updaters()
        self.wait(2)
