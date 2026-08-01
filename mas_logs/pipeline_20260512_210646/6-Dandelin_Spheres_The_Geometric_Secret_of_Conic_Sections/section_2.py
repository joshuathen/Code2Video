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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title = "Prerequisite: The 'Ice Cream' Property"
        lines = [
            "Consider a sphere and a point outside it.",
            "Tangent lines from this point are equal in length.",
            "This forms a perfect circle of contact."
        ]
        self.setup_layout(title, lines)

        # Define colors for elements matching lecture lines
        COLOR_1 = "#00FFFF" # Sphere / Line 1
        COLOR_2 = "#FFD700" # Tangents / Line 2
        COLOR_3 = "#00FF00" # Contact Circle / Line 3

        # === Animation for Lecture Line 1 ===
        # Display a sphere (#00FFFF) and a point P outside it.
        sphere = Circle(radius=1.2, color=COLOR_1, fill_opacity=0.3)
        sphere.set_fill(COLOR_1, opacity=0.3)
        self.place_in_area(sphere, "B2", "E4")
        
        p_dot = Dot(color=WHITE)
        self.place_at_grid(p_dot, "D5") # Updated per Issue 32
        
        p_label = Text("P", font_size=20, color=WHITE)
        p_label.next_to(p_dot, RIGHT, buff=0.2)
        
        sphere_label = Text("Sphere", font_size=20, color=COLOR_1)
        sphere_label.next_to(sphere, UP, buff=0.2)

        self.play(
            self.lecture[0].animate.set_color(COLOR_1),
            Create(sphere),
            FadeIn(sphere_label)
        )
        self.play(
            FadeIn(p_dot),
            FadeIn(p_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw two tangent segments from P to the sphere, ending at points T1 and T2.
        
        # Load ice cream icon asset per Issue 27
        ice_cream_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/icecream.svg")
        self.place_in_area(ice_cream_asset, "A1", "B2", scale_factor=0.6)
        
        # Geometric calculations for tangents
        center = sphere.get_center()
        p_pos = p_dot.get_center()
        radius = sphere.radius
        dist = np.linalg.norm(p_pos - center)
        
        angle_to_p = np.arctan2(p_pos[1] - center[1], p_pos[0] - center[0])
        angle_to_tangent = np.arccos(radius / dist)
        
        t1_angle = angle_to_p + angle_to_tangent
        t2_angle = angle_to_p - angle_to_tangent
        
        t1_pos = center + np.array([np.cos(t1_angle), np.sin(t1_angle), 0]) * radius
        t2_pos = center + np.array([np.cos(t2_angle), np.sin(t2_angle), 0]) * radius
        
        t1_dot = Dot(t1_pos, radius=0.05, color=COLOR_2)
        t2_dot = Dot(t2_pos, radius=0.05, color=COLOR_2)
        
        t1_label = Text("T1", font_size=18, color=COLOR_2).next_to(t1_dot, UP+LEFT, buff=0.1)
        t2_label = Text("T2", font_size=18, color=COLOR_2).next_to(t2_dot, DOWN+LEFT, buff=0.1)
        
        line_pt1 = Line(p_pos, t1_pos, color=COLOR_2)
        line_pt2 = Line(p_pos, t2_pos, color=COLOR_2)
        
        # Braces for length
        brace1 = BraceBetweenPoints(t1_pos, p_pos, color=COLOR_2, buff=0.1)
        brace2 = BraceBetweenPoints(p_pos, t2_pos, color=COLOR_2, buff=0.1)
        
        len_label1 = Text("L", font_size=18, color=COLOR_2).next_to(brace1, UP, buff=0.05)
        len_label2 = Text("L", font_size=18, color=COLOR_2).next_to(brace2, DOWN, buff=0.05)
        
        eq_text = Text("Length(PT1) = Length(PT2)", font_size=24, color=COLOR_2)
        self.place_in_area(eq_text, "F2", "F5", scale_factor=0.8) # Updated per Issue 33

        self.play(
            self.lecture[1].animate.set_color(COLOR_2),
            Create(line_pt1),
            Create(line_pt2),
            FadeIn(ice_cream_asset)
        )
        self.play(
            FadeIn(t1_dot), FadeIn(t2_dot),
            FadeIn(t1_label), FadeIn(t2_label)
        )
        self.play(
            Create(brace1), Create(brace2),
            Write(len_label1), Write(len_label2),
            FadeIn(eq_text)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This forms a perfect circle of contact.
        
        contact_center = (t1_pos + t2_pos) / 2
        major_axis = np.linalg.norm(t1_pos - t2_pos) / 2
        # Visualize the circle of contact as an ellipse passsing through T1 and T2
        contact_circle = Ellipse(
            width=0.4, 
            height=major_axis * 2, 
            color=COLOR_3, 
            stroke_width=4
        ).move_to(contact_center)
        
        # Rotate ellipse to align with the chord T1-T2
        angle_chord = np.arctan2(t1_pos[1] - t2_pos[1], t1_pos[0] - t2_pos[0])
        contact_circle.rotate(angle_chord - PI/2)

        self.play(
            self.lecture[2].animate.set_color(COLOR_3),
            Create(contact_circle)
        )
        
        # Highlight the contact area
        contact_text = Text("Circle of Contact", font_size=20, color=COLOR_3)
        self.place_in_area(contact_text, "A5", "B6", scale_factor=0.7) # Updated per Issue 34
        self.play(FadeIn(contact_text))
        
        self.wait(2)
