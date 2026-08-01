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
        # Initialization
        lines = [
            "Consider the boundary between air and dense glass.",
            "A light ray hits the boundary near the normal.",
            "Entering the glass, the ray bends toward the normal.",
            "Snell's Law mathematically predicts this change in direction.",
            "We measure these angles relative to the perpendicular normal."
        ]
        self.setup_layout("The Math: Snell's Law and Geometry", lines)
        
        # Colors
        COLOR_AIR = "#FFFFFF"
        COLOR_GLASS = "#E0FFFF"
        COLOR_LASER = "#FF0000"
        COLOR_N1 = "#FFFF00"
        COLOR_N2 = "#00FFFF"
        COLOR_NORMAL = "#FFFFFF"
        COLOR_PROTRACTOR = "#FFFFE0"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Boundary Line
        boundary = Line(self.grid["C1"], self.grid["C6"], color=WHITE)
        
        # Media regions (visual hint)
        air_rect = Rectangle(width=6, height=3, fill_opacity=0.1, fill_color=BLUE_E, stroke_width=0)
        self.place_in_area(air_rect, "A1", "B6")
        
        glass_rect = Rectangle(width=6, height=3, fill_opacity=0.3, fill_color=COLOR_GLASS, stroke_width=0)
        self.place_in_area(glass_rect, "D1", "F6")
        
        # Fixed Positioning (Issue 43 & 44)
        air_label = Text("Air (n=1.0)", font_size=18, color=COLOR_AIR)
        self.place_at_grid(air_label, "A1", scale_factor=0.8)
        
        glass_label = Text("Glass (n=1.5)", font_size=18, color=COLOR_GLASS)
        self.place_at_grid(glass_label, "F1", scale_factor=0.8)
        
        self.play(Create(boundary), FadeIn(air_rect), FadeIn(glass_rect))
        self.play(Write(air_label), Write(glass_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Asset Integration (Issue 30/57)
        laser_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/laser.svg")
        self.place_at_grid(laser_icon, "A1", scale_factor=0.3)
        laser_icon.set_color(COLOR_LASER)
        laser_icon.shift(UP * 0.3 + LEFT * 0.3)
        
        # Normal line
        normal_line = DashedLine(self.grid["A3"], self.grid["E3"], color=COLOR_NORMAL)
        normal_label = Text("Normal", font_size=14, color=COLOR_NORMAL)
        self.place_at_grid(normal_label, "A3", scale_factor=1.0)
        normal_label.shift(RIGHT * 0.6)
        
        # Incident Ray from source to hit point C3
        hit_point = self.grid["C3"]
        start_point = self.grid["A1"]
        incident_ray = Arrow(start_point, hit_point, buff=0, color=COLOR_LASER, stroke_width=4)
        
        self.play(FadeIn(laser_icon))
        self.play(Create(normal_line), Write(normal_label))
        self.play(GrowArrow(incident_ray))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Refracted Ray (calculated to bend towards normal)
        end_point = self.grid["E4"]
        refracted_ray = Arrow(hit_point, end_point, buff=0, color=COLOR_LASER, stroke_width=4)
        
        # Visual labels for angles
        incident_angle_txt = Text("Incident Angle", font_size=12, color=COLOR_N1)
        self.place_at_grid(incident_angle_txt, "B2", scale_factor=1.0)
        
        refracted_angle_txt = Text("Refracted Angle", font_size=12, color=COLOR_N2)
        self.place_at_grid(refracted_angle_txt, "D4", scale_factor=1.0)
        
        self.play(GrowArrow(refracted_ray))
        self.play(Write(incident_angle_txt), Write(refracted_angle_txt))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Formula display (Issue 42/57)
        eq_n1 = Text("n1", color=COLOR_N1, font_size=24)
        eq_sin1 = Text(" \u22c5 sin(", font_size=24)
        eq_t1 = Text("\u03b81", color=COLOR_N1, font_size=24)
        eq_close1 = Text(") = ", font_size=24)
        eq_n2 = Text("n2", color=COLOR_N2, font_size=24)
        eq_sin2 = Text(" \u22c5 sin(", font_size=24)
        eq_t2 = Text("\u03b82", color=COLOR_N2, font_size=24)
        eq_close2 = Text(")", font_size=24)
        snells_law = VGroup(eq_n1, eq_sin1, eq_t1, eq_close1, eq_n2, eq_sin2, eq_t2, eq_close2).arrange(RIGHT, buff=0.1)
        self.place_in_area(snells_law, "A4", "A6", scale_factor=0.5)
        
        self.play(Write(snells_law))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Protractor Overlay
        protractor = Arc(radius=1.5, start_angle=0, angle=-PI, arc_center=hit_point, color=COLOR_PROTRACTOR)
        protractor.set_fill(COLOR_PROTRACTOR, opacity=0.2)
        protractor_border = Line(hit_point + LEFT * 1.5, hit_point + RIGHT * 1.5, color=COLOR_PROTRACTOR)
        protractor_group = VGroup(protractor, protractor_border)
        
        # Specific angle symbols and arcs
        theta1_arc = Arc(radius=0.7, start_angle=PI/2, angle=PI/4, arc_center=hit_point, color=COLOR_N1)
        theta1_val = Text("\u03b81", font_size=16, color=COLOR_N1)
        theta1_val.move_to(hit_point + UP * 0.9 + LEFT * 0.4)
        
        theta2_angle_rad = np.radians(28.1)
        theta2_arc = Arc(radius=0.8, start_angle=-PI/2, angle=theta2_angle_rad, arc_center=hit_point, color=COLOR_N2)
        theta2_val = Text("\u03b82", font_size=16, color=COLOR_N2)
        theta2_val.move_to(hit_point + DOWN * 1.0 + RIGHT * 0.3)
        
        self.play(FadeIn(protractor_group))
        self.play(Create(theta1_arc), Write(theta1_val))
        self.play(Create(theta2_arc), Write(theta2_val))
        self.wait(2)
        
        # Final cleanup for section handover
        self.play(
            FadeOut(protractor_group), 
            FadeOut(snells_law), 
            FadeOut(theta1_arc), FadeOut(theta1_val), 
            FadeOut(theta2_arc), FadeOut(theta2_val),
            FadeOut(incident_angle_txt), FadeOut(refracted_angle_txt),
            self.lecture[4].animate.set_color(WHITE)
        )
        self.wait(1)
